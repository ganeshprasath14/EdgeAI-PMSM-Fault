# dataset_analysis_combined.py
# Generic EDA for three-phase current OR vibration data.
# Automatically detects sensor type (Current / Vibration) from the folder name.
# Uses a configuration helper to avoid hardcoded sensor‑specific logic throughout.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile
from scipy import stats


# ---------------------------------------------------------------------
# Sensor configuration helper – centralizes all sensor‑specific settings
# ---------------------------------------------------------------------
def sensor_config(sensor_type):
    """
    Return a dictionary with labels, titles, and folder names for the given sensor.
    This avoids repeated `if sensor_type == ...` checks in the main code.
    """
    if sensor_type == "Current":
        return {
            "signal_name": "Current",
            "title_signal": "Three‑Phase Currents",
            "axis_label": "Current (A)",          # we know this is Amperes
            "table_unit": "A",
            "save_folder": "Current",
            "channel_labels": {
                "cDAQ1Mod2/ai0": "Phase A",
                "cDAQ1Mod2/ai2": "Phase B",
                "cDAQ1Mod2/ai3": "Phase C",
            },
            "color_palette": ['blue', 'orange', 'green'],
            "col_header": "Phase"                 # column name in statistics table
        }
    else:  # Vibration – no unit assumed
        return {
            "signal_name": "Vibration",
            "title_signal": "Vibration Signal",
            "axis_label": "Amplitude",            # no unit, generic
            "table_unit": "",                     # empty unit
            "save_folder": "Vibration",
            "channel_labels": {},                 # will be built dynamically
            "color_palette": ['blue'],
            "col_header": "Channel"               # column name in statistics table
        }


# ---------------------------------------------------------------------
# File information parser
# ---------------------------------------------------------------------
def parse_file_info(file_path):
    """
    Extract motor rating, fault type, severity, dataset family, and sensor type.
    Returns: (motor_rating, fault_type, severity, dataset_family, sensor_type)
    """
    parts = file_path.split(os.sep)
    motor_rating = "Unknown"
    sensor_type = "Unknown"

    # Detect sensor type and motor rating from folder structure
    for i, part in enumerate(parts):
        if part.lower() in ["current", "vibration"] and i > 0:
            sensor_type = part.capitalize()   # "Current" or "Vibration"
            motor_rating = parts[i-1]         # e.g., "1.0kW"
            break

    filename = os.path.splitext(parts[-1])[0]
    name_lower = filename.lower()

    # ---- Determine fault type ----
    if "interturn" in name_lower:
        if "_0_00_" in name_lower:
            fault_type = "Healthy"
        else:
            fault_type = "ITSC"
    elif "coil" in name_lower or "intercoil" in name_lower or "ccsc" in name_lower:
        if "_0_00_" in name_lower:
            fault_type = "Healthy"
        else:
            fault_type = "CCSC"
    else:
        fault_type = "Unknown"

    # ---- Determine dataset family (ITSC or CCSC) ----
    if "interturn" in name_lower:
        dataset_family = "ITSC"
    elif "coil" in name_lower or "intercoil" in name_lower or "ccsc" in name_lower:
        dataset_family = "CCSC"
    else:
        dataset_family = "Unknown"

    # ---- Extract severity (numeric parts) ----
    number_parts = [p for p in filename.split('_') if p.isdigit()]
    if len(number_parts) >= 2:
        severity = f"{number_parts[0]}_{number_parts[1]}"
    elif len(number_parts) == 1:
        severity = number_parts[0]
    else:
        severity = "Unknown"

    return motor_rating, fault_type, severity, dataset_family, sensor_type


# ---------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------
def main():
    # ---- UNCOMMENT THE DESIRED FILE PATH ----
    # Current datasets (folder 'current')
    # Healthy ITSC:
    # file_path = r"C:\FinalYearProject\Dataset\1.0kW\current\1000W_0_00_current_interturn.tdms"
    # ITSC fault:
    # file_path = r"C:\FinalYearProject\Dataset\1.0kW\current\1000W_2_26_current_interturn.tdms"
    # Healthy CCSC:
    # file_path = r"C:\FinalYearProject\Dataset\1.0kW\current\1000W_0_00_current_coil.tdms"
    # CCSC fault:
    file_path = r"C:\FinalYearProject\Dataset\1.0kW\current\1000W_0_68_current_coil.tdms"

    # Vibration datasets (folder 'vibration')
    # Healthy ITSC:
    # file_path = r"C:\FinalYearProject\Dataset\1.0kW\vibration\1000W_0_00_vibration_interturn.tdms"
    # ITSC fault:
    # file_path = r"C:\FinalYearProject\Dataset\1.0kW\vibration\1000W_2_26_vibration_interturn.tdms"
    # Healthy CCSC:
    # file_path = r"C:\FinalYearProject\Dataset\1.0kW\vibration\1000W_0_00_vibration_coil.tdms"
    # CCSC fault:
    file_path = r"C:\FinalYearProject\Dataset\1.0kW\vibration\1000W_0_68_vibration_coil.tdms"

    start_time = time.time()

    motor_rating, fault_type, fault_severity, dataset_family, sensor_type = parse_file_info(file_path)
    cfg = sensor_config(sensor_type)   # get all sensor‑specific settings

    # Read TDMS
    try:
        tdms_file = TdmsFile.read(file_path)
    except Exception as e:
        print("Error reading TDMS file:", e)
        return

    groups = tdms_file.groups()
    if len(groups) == 0:
        raise ValueError("No groups found.")

    # Find first group with channels
    group = None
    for g in groups:
        if len(g.channels()) > 0:
            group = g
            break
    if group is None:
        raise ValueError("No channels found.")

    channels = group.channels()
    print(f"Selected Group: {group.name}")
    print(f"Available channels: {[ch.name for ch in channels]}")

    # ---- Build channel labels based on sensor type ----
    if sensor_type == "Current":
        # Use predefined mapping for current phases
        def get_label(raw_name):
            return cfg["channel_labels"].get(raw_name, raw_name)
    else:  # Vibration
        # For vibration, produce "Vibration 1", "Vibration 2", ...
        # If only one channel, just "Vibration"
        num_channels = len(channels)
        if num_channels == 1:
            def get_label(raw_name):
                return "Vibration"
        else:
            # Build a mapping dynamically
            label_map = {ch.name: f"Vibration {i+1}" for i, ch in enumerate(channels)}
            def get_label(raw_name):
                return label_map.get(raw_name, raw_name)

    # We'll store data for all channels
    channel_names = []
    signals = []
    sampling_rate = None
    duration = None
    time_axis = None

    # First pass: read signals and verify sampling rate consistency
    for ch in channels:
        signal = ch[:]
        signals.append(signal)
        channel_names.append(ch.name)

        si = ch.properties.get("wf_increment")
        if si is None:
            raise KeyError(f"wf_increment property missing for channel {ch.name}")
        current_sr = 1.0 / si

        if sampling_rate is None:
            sampling_rate = current_sr
            duration = len(signal) * si
            time_axis = np.arange(len(signal)) * si
        else:
            if abs(current_sr - sampling_rate) > 1e-6:
                raise ValueError(
                    f"Sampling rate mismatch: {sampling_rate:.2f} Hz vs {current_sr:.2f} Hz "
                    f"for channel {ch.name}"
                )

    # ================================================================
    # Compute statistics for each channel
    # ================================================================
    stats_dict = {}
    for name, sig in zip(channel_names, signals):
        stats_dict[name] = {
            "Min": np.min(sig),
            "Max": np.max(sig),
            "Mean": np.mean(sig),
            "Std": np.std(sig),
            "RMS": np.sqrt(np.mean(sig**2)),
            "Peak": np.max(np.abs(sig)),
            "Skew": stats.skew(sig),
            "Kurtosis": stats.kurtosis(sig),
        }

    # ================================================================
    # Print consolidated report
    # ================================================================
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    print("\n" + "="*70)
    print(f"         CONSOLIDATED TDMS ANALYSIS – {sensor_type.upper()} SIGNAL")
    print("="*70)
    print(f"File Path        : {file_path}")
    print(f"File Size        : {file_size_mb:.2f} MB")
    print(f"Motor Rating     : {motor_rating}")
    print(f"Dataset Family   : {dataset_family}")
    print(f"Fault Type       : {fault_type}")
    print(f"Fault Severity   : {fault_severity}")
    print(f"Sampling Rate    : {sampling_rate:.0f} Hz")
    print(f"Duration         : {duration:.4f} s")
    print(f"Number of Channels: {len(channels)}")
    print("-"*70)

    # ---- Signal Statistics header ----
    # Use generic header without assuming units; column label is "Phase" or "Channel"
    unit = cfg.get("table_unit", "")
    if unit:
        unit_str = f" ({unit})"
    else:
        unit_str = ""
    col_label = cfg["col_header"]   # "Phase" or "Channel"
    header = (f"{col_label:<12} {'Mean':>10}{unit_str} {'Std':>10}{unit_str} "
              f"{'RMS':>10}{unit_str} {'Peak':>10}{unit_str} "
              f"{'Min':>10}{unit_str} {'Max':>10}{unit_str} {'Skew':>8} {'Kurtosis':>10}")
    print("SIGNAL STATISTICS")
    print(header)
    print("-"*70)

    for raw_name in channel_names:
        label = get_label(raw_name)
        s = stats_dict[raw_name]
        # Print values with consistent formatting
        print(f"{label:<12} {s['Mean']:10.4f} {s['Std']:10.4f} {s['RMS']:10.4f} "
              f"{s['Peak']:10.4f} {s['Min']:10.4f} {s['Max']:10.4f} "
              f"{s['Skew']:8.3f} {s['Kurtosis']:10.3f}")
    print("="*70)

    # ================================================================
    # Combined Histogram
    # ================================================================
    display_status = "Healthy" if fault_type == "Healthy" else "Fault"
    plot_labels = [get_label(name) for name in channel_names]

    plt.figure(figsize=(10, 6))
    colors = cfg["color_palette"]
    # If more channels than colors, cycle
    for i, (label, sig) in enumerate(zip(plot_labels, signals)):
        plt.hist(sig, bins=100, alpha=0.5, color=colors[i % len(colors)],
                 label=label, density=True)
    # Title
    plt.title(f"{cfg['signal_name']} Distribution – {motor_rating} | {dataset_family} Dataset | {display_status} | Severity {fault_severity}\n"
              f"Sampling Frequency = {sampling_rate:.0f} Hz")
    plt.xlabel(cfg["axis_label"])
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Save path
    hist_path = os.path.join(
        r"C:\FinalYearProject\Figures",
        cfg["save_folder"],
        "Histogram",
        f"histogram_combined_{motor_rating}_{dataset_family}_{fault_type}_{fault_severity}.png"
    )
    os.makedirs(os.path.dirname(hist_path), exist_ok=True)
    plt.savefig(hist_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nCombined histogram saved: {hist_path}")

    # ================================================================
    # Combined Time‑Domain Zoom (first 1000 samples)
    # ================================================================
    n_samples = 1000
    plt.figure(figsize=(12, 5))
    for i, (label, sig) in enumerate(zip(plot_labels, signals)):
        plt.plot(time_axis[:n_samples], sig[:n_samples],
                 color=colors[i % len(colors)], linewidth=0.8, label=label)

    plt.title(f"{cfg['title_signal']} – {motor_rating} | {dataset_family} Dataset | {display_status} | Severity {fault_severity}\n"
              f"First {n_samples} samples (Sampling Rate = {sampling_rate:.0f} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel(cfg["axis_label"])
    plt.legend()
    plt.grid(True, alpha=0.3)

    zoom_path = os.path.join(
        r"C:\FinalYearProject\Figures",
        cfg["save_folder"],
        "TimeDomain",
        f"zoom_combined_{n_samples}samples_{motor_rating}_{dataset_family}_{fault_type}_{fault_severity}.png"
    )
    os.makedirs(os.path.dirname(zoom_path), exist_ok=True)
    plt.savefig(zoom_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Combined time‑domain plot saved: {zoom_path}")

    # ================================================================
    # Execution time
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()