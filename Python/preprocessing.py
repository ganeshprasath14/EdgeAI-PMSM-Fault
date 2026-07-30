# preprocessing.py
# =================
# Final preprocessing pipeline (v1.3)
# Features:
#   - Automatic discovery of all .tdms files
#   - DC offset removal + z‑score normalisation
#   - Windowing (1024 samples, 50 % overlap)
#   - Separate storage per file (Current/ Vibration) to avoid memory overflow
#   - Traceable window labels (source_file, window_id, ...)
#   - Comprehensive configuration file (JSON)
#   - Detailed per‑file preprocessing statistics (CSV)
#   - Log file for skipped/corrupted files
#   - Memory freed after each file (del, gc.collect())
#   - Per‑file .npz saved immediately → no loss on interruption
#   - Unit tests (optional)
#
# Changes v1.2 -> v1.3 (major redesign):
#   - No more accumulating all windows in RAM
#   - Each TDMS file → windows → immediate save to disk
#   - Output structure: ProcessedDataset/Current/<filename>.npz, ProcessedDataset/Vibration/<filename>.npz
#   - Memory freed after each file with del and gc.collect()
#   - Total window counts updated during processing
#   - Summary CSV updated with per‑file info (including saved file path)
#
# Additional minor updates:
#   - Replaced np.savez_compressed with np.savez (faster loading, negligible space difference)
#   - Improved exception handling: check variable existence before del, avoid bare except

import os
import csv
import json
import time
import logging
import argparse
import sys
import gc
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
from nptdms import TdmsFile


# =============================================================================
# CONSTANTS
# =============================================================================
SENSOR_CURRENT = "Current"
SENSOR_VIBRATION = "Vibration"
FAULT_HEALTHY = "Healthy"
FAULT_ITSC = "ITSC"
FAULT_CCSC = "CCSC"
DATASET_FAMILY_ITSC = "ITSC"
DATASET_FAMILY_CCSC = "CCSC"
UNKNOWN = "Unknown"

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_ROOT = r"C:\FinalYearProject\Dataset"
PROCESSED_ROOT = os.path.join(DATASET_ROOT, "ProcessedDataset")

WINDOW_SIZE = 1024
OVERLAP = 0.5
DATASET_VERSION = "v1.3"
NORMALIZATION = "z-score"
DC_OFFSET_REMOVAL = True
DTYPE = np.float32

# =============================================================================
# LOGGING SETUP
# =============================================================================
def setup_logging(log_dir: str) -> logging.Logger:
    """Configure logging to both console and a file."""
    logger = logging.getLogger("Preprocessing")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_file = os.path.join(log_dir, "preprocessing.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


# =============================================================================
# SENSOR CONFIGURATION HELPER
# =============================================================================
def sensor_config(sensor_type: str) -> Dict[str, Any]:
    """Return a configuration dictionary for a given sensor type."""
    if sensor_type == SENSOR_CURRENT:
        return {
            "signal_name": SENSOR_CURRENT,
            "channel_labels": {
                "cDAQ1Mod2/ai0": "Phase A",
                "cDAQ1Mod2/ai2": "Phase B",
                "cDAQ1Mod2/ai3": "Phase C",
            },
        }
    else:
        return {
            "signal_name": SENSOR_VIBRATION,
            "channel_labels": {},
        }


# =============================================================================
# FILE INFORMATION PARSER
# =============================================================================
def parse_file_info(file_path: str) -> Tuple[str, str, str, str, str]:
    """
    Parse motor rating, fault type, severity, dataset family, and sensor type
    from the file path and filename.
    """
    parts = file_path.split(os.sep)
    motor_rating = UNKNOWN
    sensor_type = UNKNOWN

    for i, part in enumerate(parts):
        if part.lower() in ["current", "vibration"] and i > 0:
            sensor_type = part.capitalize()
            motor_rating = parts[i - 1]
            break

    filename = os.path.splitext(parts[-1])[0]
    name_lower = filename.lower()

    if "interturn" in name_lower:
        if "_0_00_" in name_lower:
            fault_type = FAULT_HEALTHY
        else:
            fault_type = FAULT_ITSC
    elif "coil" in name_lower or "intercoil" in name_lower or "ccsc" in name_lower:
        if "_0_00_" in name_lower:
            fault_type = FAULT_HEALTHY
        else:
            fault_type = FAULT_CCSC
    else:
        fault_type = UNKNOWN

    if "interturn" in name_lower:
        dataset_family = DATASET_FAMILY_ITSC
    elif "coil" in name_lower or "intercoil" in name_lower or "ccsc" in name_lower:
        dataset_family = DATASET_FAMILY_CCSC
    else:
        dataset_family = UNKNOWN

    number_parts = [p for p in filename.split('_') if p.isdigit()]
    if len(number_parts) >= 2:
        severity = f"{number_parts[0]}_{number_parts[1]}"
    elif len(number_parts) == 1:
        severity = number_parts[0]
    else:
        severity = UNKNOWN

    return motor_rating, fault_type, severity, dataset_family, sensor_type


# =============================================================================
# FILE DISCOVERY
# =============================================================================
def find_tdms_files(root_dir: str) -> List[str]:
    """Recursively find all .tdms files under root_dir."""
    tdms_files: List[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith('.tdms'):
                tdms_files.append(os.path.join(dirpath, f))
    tdms_files.sort()
    return tdms_files


# =============================================================================
# TDMS READER
# =============================================================================
def read_tdms(file_path: str) -> Tuple[np.ndarray, float, List[str]]:
    """
    Read a TDMS file and extract signals, sampling rate, and channel names.
    Returns (signals, sampling_rate, channel_names)
    """
    try:
        tdms_file = TdmsFile.read(file_path)
    except Exception as e:
        raise IOError(f"Error reading TDMS file: {file_path}") from e

    groups = tdms_file.groups()
    if not groups:
        raise ValueError("No groups found.")

    group = None
    for g in groups:
        if len(g.channels()) > 0:
            group = g
            break
    if group is None:
        raise ValueError("No channels found.")

    channels = group.channels()
    signals = []
    channel_names = []
    sampling_rate = None

    for ch in channels:
        sig = ch[:]
        signals.append(sig)
        channel_names.append(ch.name)

        si = ch.properties.get("wf_increment")
        if si is None:
            raise KeyError(f"wf_increment property missing for channel {ch.name}")
        current_sr = 1.0 / si

        if sampling_rate is None:
            sampling_rate = current_sr
        elif abs(current_sr - sampling_rate) > 1e-6:
            raise ValueError(f"Sampling rate mismatch in {file_path}.")

    signals = np.array(signals).T
    return signals, float(sampling_rate), channel_names


# =============================================================================
# PREPROCESSING FUNCTIONS
# =============================================================================
def remove_dc_offset(signals: np.ndarray) -> np.ndarray:
    """Remove DC offset by subtracting the mean along the time axis."""
    if signals.ndim == 1:
        return signals - np.mean(signals)
    return signals - np.mean(signals, axis=0, keepdims=True)


def normalize_signal(signals: np.ndarray) -> np.ndarray:
    """Apply z‑score normalisation (zero mean, unit variance) per channel."""
    if signals.ndim == 1:
        std = np.std(signals)
        if std == 0:
            return signals
        return (signals - np.mean(signals)) / std
    else:
        mean = np.mean(signals, axis=0, keepdims=True)
        std = np.std(signals, axis=0, keepdims=True)
        std[std == 0] = 1.0
        return (signals - mean) / std


def create_windows(signals: np.ndarray,
                   window_size: int = WINDOW_SIZE,
                   overlap: float = OVERLAP) -> np.ndarray:
    """
    Split a continuous signal into overlapping windows.
    Returns array of shape (num_windows, window_size, n_channels).
    """
    if not (0 <= overlap < 1):
        raise ValueError("Overlap must be between 0 and 1 (inclusive 0, exclusive 1).")
    step = int(window_size * (1 - overlap))
    n_samples = signals.shape[0]
    windows = []
    start = 0
    while start + window_size <= n_samples:
        window = signals[start:start + window_size, :]
        windows.append(window)
        start += step
    if not windows:
        return np.array([])
    return np.stack(windows, axis=0)


# =============================================================================
# PER‑FILE SAVER
# =============================================================================
def save_windows_to_file(windows: np.ndarray,
                         labels: List[Dict[str, Any]],
                         output_path: str,
                         dtype: np.dtype = DTYPE) -> None:
    """
    Save windows and their labels to a single .npz file.
    The file can later be loaded with np.load().
    Uses uncompressed .npz for faster loading (negligible space saving with float32).
    """
    if windows.size == 0:
        print(f"No windows to save for {output_path}")
        return
    # Convert to specified dtype to save space
    windows_array = windows.astype(dtype)
    labels_array = np.array(labels, dtype=object)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Using savez (not compressed) for faster I/O
    np.savez(output_path, windows=windows_array, labels=labels_array)
    print(f"  Saved {windows_array.shape[0]} windows to {output_path}")


# =============================================================================
# GENERATE OUTPUT FILENAME
# =============================================================================
def generate_output_filename(original_filename: str, sensor_type: str) -> str:
    """Create a unique .npz filename for the given sensor."""
    # Remove .tdms extension
    base = os.path.splitext(original_filename)[0]
    # Append sensor type to avoid collisions if same base name for both sensors
    return f"{base}_{sensor_type}.npz"


# =============================================================================
# CONFIGURATION SAVER
# =============================================================================
def save_preprocessing_config(output_dir: str,
                              total_current_windows: int,
                              total_vibration_windows: int,
                              processed_files: int,
                              skipped_files: int) -> None:
    """Save preprocessing parameters and statistics as JSON."""
    config = {
        "dataset_version": DATASET_VERSION,
        "window_size": WINDOW_SIZE,
        "overlap": OVERLAP,
        "normalization": NORMALIZATION,
        "dc_offset_removal": DC_OFFSET_REMOVAL,
        "dataset_root": DATASET_ROOT,
        "processed_root": PROCESSED_ROOT,
        "dtype": str(DTYPE),
        "total_current_windows": total_current_windows,
        "total_vibration_windows": total_vibration_windows,
        "processed_files": processed_files,
        "skipped_files": skipped_files,
    }
    config_path = os.path.join(output_dir, "preprocessing_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Configuration saved to {config_path}")


# =============================================================================
# UNIT TESTS (optional)
# =============================================================================
def run_tests() -> bool:
    """Run a suite of unit tests for the helper functions."""
    print("Running unit tests...")
    all_passed = True

    # Test remove_dc_offset
    sig = np.array([1.0, 2.0, 3.0])
    out = remove_dc_offset(sig)
    np.testing.assert_almost_equal(out, [-1.0, 0.0, 1.0], decimal=6)
    sig2d = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    out2d = remove_dc_offset(sig2d)
    expected = np.array([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])
    np.testing.assert_almost_equal(out2d, expected, decimal=6)

    # Test normalize_signal
    sig = np.array([0.0, 0.0, 0.0])
    out = normalize_signal(sig)
    np.testing.assert_almost_equal(out, [0.0, 0.0, 0.0])
    sig2 = np.array([1.0, 2.0, 3.0])
    out2 = normalize_signal(sig2)
    np.testing.assert_almost_equal(np.mean(out2), 0.0, decimal=6)
    np.testing.assert_almost_equal(np.std(out2), 1.0, decimal=6)

    # Test create_windows
    sig = np.arange(100).reshape(-1, 1)
    win = create_windows(sig, window_size=10, overlap=0.5)
    assert win.shape == (19, 10, 1)
    assert win[0, 0, 0] == 0
    assert win[1, 0, 0] == 5
    win2 = create_windows(sig, window_size=200, overlap=0.5)
    assert win2.size == 0
    try:
        create_windows(sig, window_size=10, overlap=1.0)
    except ValueError:
        pass
    else:
        all_passed = False
        print("Test create_windows overlap validation failed")

    print("All tests passed." if all_passed else "Some tests failed.")
    return all_passed


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main() -> None:
    """Main preprocessing pipeline entry point."""
    parser = argparse.ArgumentParser(description="Preprocess TDMS files.")
    parser.add_argument("--test", action="store_true", help="Run unit tests and exit.")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    start_time = time.time()

    # 1. Create folder structure
    os.makedirs(PROCESSED_ROOT, exist_ok=True)
    current_dir = os.path.join(PROCESSED_ROOT, SENSOR_CURRENT)
    vibration_dir = os.path.join(PROCESSED_ROOT, SENSOR_VIBRATION)
    logs_dir = os.path.join(PROCESSED_ROOT, "Logs")
    config_dir = os.path.join(PROCESSED_ROOT, "Config")
    for d in [current_dir, vibration_dir, logs_dir, config_dir]:
        os.makedirs(d, exist_ok=True)

    # Setup logging
    logger = setup_logging(logs_dir)
    logger.info("=" * 60)
    logger.info("STARTING PREPROCESSING PIPELINE (v1.3 – per‑file saving)")
    logger.info("=" * 60)

    # 2. Find all TDMS files
    tdms_files = find_tdms_files(DATASET_ROOT)
    n_total = len(tdms_files)
    logger.info(f"Found {n_total} TDMS files.")

    # 3. Prepare logs and stats files
    skipped_log_path = os.path.join(logs_dir, "skipped_files.txt")
    stats_csv_path = os.path.join(logs_dir, "preprocessing_stats.csv")

    with open(skipped_log_path, "w", encoding="utf-8") as skipped_file, \
         open(stats_csv_path, "w", newline="", encoding="utf-8") as stats_file:

        skipped_file.write("Skipped files (errors or too short):\n")
        stats_writer = csv.writer(stats_file)
        stats_writer.writerow(["Filename", "Channel", "BeforeDC_Mean", "BeforeDC_Std",
                               "AfterNorm_Mean", "AfterNorm_Std"])

        # 4. Data collectors (only for summaries, not for storing windows)
        csv_rows: List[Dict[str, Any]] = []          # for summary CSV
        total_current_windows = 0
        total_vibration_windows = 0
        current_file_count = 0
        vibration_file_count = 0
        window_count_by_fault = {FAULT_HEALTHY: 0, FAULT_ITSC: 0, FAULT_CCSC: 0}
        processed_count = 0
        skipped_count = 0

        # 5. Process each file with progress indicator
        for idx, file_path in enumerate(tdms_files, 1):
            filename = os.path.basename(file_path)
            logger.info(f"\n[{idx}/{n_total}] Processing {filename}")

            # Ensure variables are defined for cleanup block
            signals = None
            windows = None
            labels = None

            try:
                motor_rating, fault_type, severity, dataset_family, sensor_type = parse_file_info(file_path)

                signals, sr, channel_names = read_tdms(file_path)
                file_duration = signals.shape[0] / sr
                num_channels = signals.shape[1]
                logger.info(f"  Channels: {num_channels}, Duration: {file_duration:.2f}s, SR: {sr:.0f} Hz")

                # Check for NaN / Inf
                if np.isnan(signals).any():
                    logger.warning(f"  WARNING: NaN values detected in {filename}")
                if np.isinf(signals).any():
                    logger.warning(f"  WARNING: Inf values detected in {filename}")

                # Verify channel count based on sensor type
                if sensor_type == SENSOR_CURRENT and num_channels != 3:
                    raise ValueError(f"Expected 3 channels for Current, got {num_channels}")
                if sensor_type == SENSOR_VIBRATION and num_channels != 1:
                    raise ValueError(f"Expected 1 channel for Vibration, got {num_channels}")

                # Statistics before DC removal
                before_means = np.mean(signals, axis=0)
                before_stds = np.std(signals, axis=0)

                if DC_OFFSET_REMOVAL:
                    signals = remove_dc_offset(signals)

                signals = normalize_signal(signals)

                after_means = np.mean(signals, axis=0)
                after_stds = np.std(signals, axis=0)

                # Write per‑channel stats
                for j, ch_name in enumerate(channel_names):
                    stats_writer.writerow([
                        filename,
                        ch_name,
                        f"{before_means[j]:.6f}",
                        f"{before_stds[j]:.6f}",
                        f"{after_means[j]:.6f}",
                        f"{after_stds[j]:.6f}"
                    ])

                # Windowing
                windows = create_windows(signals, window_size=WINDOW_SIZE, overlap=OVERLAP)
                if windows.size == 0:
                    msg = f"Signal too short for windowing: {file_path}"
                    logger.warning(f"  {msg}")
                    skipped_file.write(msg + "\n")
                    skipped_count += 1
                    # Clean up and continue
                    if signals is not None:
                        del signals
                    if windows is not None:
                        del windows
                    gc.collect()
                    continue

                num_windows = windows.shape[0]
                logger.info(f"  Windows generated: {num_windows}")

                # Update counters
                if sensor_type == SENSOR_CURRENT:
                    current_file_count += 1
                    total_current_windows += num_windows
                else:
                    vibration_file_count += 1
                    total_vibration_windows += num_windows

                if fault_type in window_count_by_fault:
                    window_count_by_fault[fault_type] += num_windows
                else:
                    window_count_by_fault[fault_type] = num_windows

                # Build labels for each window
                labels = []
                for i in range(num_windows):
                    label = {
                        'source_file': filename,
                        'window_id': i,
                        'motor_rating': motor_rating,
                        'fault_type': fault_type,
                        'severity': severity,
                        'dataset_family': dataset_family,
                        'sensor_type': sensor_type,
                        'sampling_rate': sr,
                        'window_size': WINDOW_SIZE,
                        'overlap': OVERLAP,
                        'file_duration': file_duration,
                        'num_channels': num_channels,
                    }
                    labels.append(label)

                # Determine output directory and filename
                if sensor_type == SENSOR_CURRENT:
                    out_dir = current_dir
                else:
                    out_dir = vibration_dir

                out_filename = generate_output_filename(filename, sensor_type)
                out_path = os.path.join(out_dir, out_filename)

                # Save windows and labels immediately
                save_windows_to_file(windows, labels, out_path, dtype=DTYPE)

                # Append to CSV summary row (with saved file path)
                csv_rows.append({
                    "Filename": filename,
                    "Motor": motor_rating,
                    "Sensor": sensor_type,
                    "Fault": fault_type,
                    "Windows": num_windows,
                    "SamplingRate": f"{sr:.0f}",
                    "Duration_s": f"{file_duration:.2f}",
                    "WindowSize": WINDOW_SIZE,
                    "Overlap": OVERLAP,
                    "SavedFile": out_filename,
                })

                processed_count += 1

                # Free memory before next file
                if signals is not None:
                    del signals
                if windows is not None:
                    del windows
                if labels is not None:
                    del labels
                gc.collect()

                # (Optional) Print RAM usage if needed
                # import psutil; print(f"RAM: {psutil.Process().memory_info().rss / 1e9:.2f} GB")

            except Exception as e:
                error_msg = f"Skipping {filename}: {e}"
                logger.warning(f"  {error_msg}")
                skipped_file.write(error_msg + "\n")
                skipped_count += 1
                # Clean up any partially loaded data
                if signals is not None:
                    del signals
                if windows is not None:
                    del windows
                if labels is not None:
                    del labels
                gc.collect()
                continue

    # After 'with' block, files are closed.

    # 6. Write CSV summary
    csv_path = os.path.join(PROCESSED_ROOT, "preprocessing_summary.csv")
    # Updated fieldnames
    fieldnames = ["Filename", "Motor", "Sensor", "Fault", "Windows",
                  "SamplingRate", "Duration_s", "WindowSize", "Overlap", "SavedFile"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"\nCSV summary saved: {csv_path}")

    # 7. Save configuration (with totals)
    save_preprocessing_config(
        config_dir,
        total_current_windows,
        total_vibration_windows,
        processed_files=processed_count,
        skipped_files=skipped_count
    )

    # 8. Final summary
    elapsed = time.time() - start_time
    total_files = current_file_count + vibration_file_count
    logger.info("\n" + "=" * 60)
    logger.info("               PREPROCESSING COMPLETED")
    logger.info("=" * 60)
    logger.info(f"  Dataset version   : {DATASET_VERSION}")
    logger.info(f"  Processed files   : {processed_count} (Current: {current_file_count}, Vibration: {vibration_file_count})")
    logger.info(f"  Skipped files     : {skipped_count} (see Logs/skipped_files.txt)")
    logger.info("  Generated windows :")
    for fault in sorted(window_count_by_fault.keys()):
        logger.info(f"    {fault:<10} : {window_count_by_fault[fault]}")
    logger.info("-" * 60)
    logger.info(f"  Total Current windows  : {total_current_windows}")
    logger.info(f"  Total Vibration windows: {total_vibration_windows}")
    logger.info(f"  Data type (storage)    : {DTYPE}")
    logger.info(f"  Execution time         : {elapsed:.1f} seconds")
    logger.info("=" * 60)
    logger.info(f"\nSkipped files log: {skipped_log_path}")
    logger.info(f"Preprocessing log: {os.path.join(logs_dir, 'preprocessing.log')}")
    logger.info(f"Processed files saved in: {current_dir} and {vibration_dir}")


if __name__ == "__main__":
    main()