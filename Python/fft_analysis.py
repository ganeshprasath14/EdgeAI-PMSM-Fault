# fft_analysis.py
# Frequency-domain analysis (FFT) for current or vibration data.
# Reuses the same sensor detection and file parsing logic as the EDA script.
# Computes and plots the single-sided amplitude spectrum (limited to a display range),
# prints key frequency metrics, and saves the figure under Figures/<sensor>/FFT/.

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile


# ---------------------------------------------------------------------
# Sensor configuration helper – centralizes all sensor‑specific settings
# ---------------------------------------------------------------------
def sensor_config(sensor_type):
    """
    Return a dictionary with labels, titles, and folder names for the given sensor.
    """
    if sensor_type == "Current":
        return {
            "signal_name": "Current",
            "title_signal": "Three‑Phase Currents",
            "axis_label": "Current (A)",          # for time domain (not used here)
            "fft_ylabel": "Amplitude",
            "save_folder": "Current",
            "channel_labels": {
                "cDAQ1Mod2/ai0": "Phase A",
                "cDAQ1Mod2/ai2": "Phase B",
                "cDAQ1Mod2/ai3": "Phase C",
            },
            "color_palette": ['blue', 'orange', 'green'],
            "col_header": "Phase",
            "max_display_freq": 50000             # full range up to Nyquist (50 kHz)
        }
    else:  # Vibration – no unit assumed
        return {
            "signal_name": "Vibration",
            "title_signal": "Vibration Signal",
            "axis_label": "Amplitude",
            "fft_ylabel": "Amplitude",
            "save_folder": "Vibration",
            "channel_labels": {},
            "color_palette": ['blue'],
            "col_header": "Channel",
            "max_display_freq": 5000              # keep 5 kHz for vibration
        }


# ---------------------------------------------------------------------
# File information parser (identical to EDA version)
# ---------------------------------------------------------------------
def parse_file_info(file_path):
    """
    Extract motor rating, fault type, severity, dataset family, and sensor type.
    Returns: (motor_rating, fault_type, severity, dataset_family, sensor_type)
    """
    parts = file_path.split(os.sep)
    motor_rating = "Unknown"
    sensor_type = "Unknown"

    for i, part in enumerate(parts):
        if part.lower() in ["current", "vibration"] and i > 0:
            sensor_type = part.capitalize()
            motor_rating = parts[i-1]
            break

    filename = os.path.splitext(parts[-1])[0]
    name_lower = filename.lower()

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

    if "interturn" in name_lower:
        dataset_family = "ITSC"
    elif "coil" in name_lower or "intercoil" in name_lower or "ccsc" in name_lower:
        dataset_family = "CCSC"
    else:
        dataset_family = "Unknown"

    number_parts = [p for p in filename.split('_') if p.isdigit()]
    if len(number_parts) >= 2:
        severity = f"{number_parts[0]}_{number_parts[1]}"
    elif len(number_parts) == 1:
        severity = number_parts[0]
    else:
        severity = "Unknown"

    return motor_rating, fault_type, severity, dataset_family, sensor_type


# ---------------------------------------------------------------------
# FFT computation helper (correct single-sided scaling)
# ---------------------------------------------------------------------
def compute_fft(signal, sampling_rate):
    """
    Compute the single-sided amplitude spectrum with correct scaling.

    Parameters
    ----------
    signal : ndarray
        Time-domain signal (real-valued).
    sampling_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    frequency : ndarray
        Frequency axis (Hz).
    amplitude : ndarray
        Single-sided amplitude spectrum.
    """
    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(n, d=1.0 / sampling_rate)

    # Correct single-sided amplitude scaling
    amplitude = np.abs(fft_vals) / n
    # Do not double the DC and Nyquist bins (they are unique)
    amplitude[1:-1] *= 2   # double all bins except DC and Nyquist

    return freq, amplitude


# ---------------------------------------------------------------------
# FFT plotting helper
# ---------------------------------------------------------------------
def plot_fft(frequencies_list, amplitudes_list, labels, cfg, motor_rating,
             dataset_family, fault_type, fault_severity, sampling_rate,
             max_display_freq, save_path):
    """
    Plot the frequency spectra of multiple signals on the same axes,
    limited to a specified display frequency range.

    Parameters
    ----------
    frequencies_list : list of ndarray
        Frequency axes for each signal.
    amplitudes_list : list of ndarray
        Amplitude spectra for each signal.
    labels : list of str
        Labels for the legend.
    cfg : dict
        Sensor configuration dictionary.
    motor_rating, dataset_family, fault_type, fault_severity : str
        Metadata for the title.
    sampling_rate : float
        Sampling frequency (Hz).
    max_display_freq : float
        Upper limit of frequency axis (Hz).
    save_path : str
        Full path where the figure will be saved.
    """
    plt.figure(figsize=(12, 5))
    colors = cfg["color_palette"]

    for i, (freq, amp) in enumerate(zip(frequencies_list, amplitudes_list)):
        mask = freq <= max_display_freq
        plt.semilogy(freq[mask], amp[mask],
                     color=colors[i % len(colors)],
                     linewidth=0.8, label=labels[i])

    display_status = "Healthy" if fault_type == "Healthy" else "Fault"
    # Include sensor type in the title
    title = (f"Single-Sided Amplitude Spectrum ({cfg['signal_name']}) – "
             f"{motor_rating} | {dataset_family} Dataset | "
             f"{display_status} | Severity {fault_severity}\n"
             f"Sampling Frequency = {sampling_rate:.0f} Hz")
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(cfg["fft_ylabel"])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(0, max_display_freq)   # enforce the display range
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"FFT plot saved: {save_path}")


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
    # file_path = r"C:\FinalYearProject\Dataset\1.0kW\current\1000W_0_68_current_coil.tdms"

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
    cfg = sensor_config(sensor_type)

    # Read TDMS
    try:
        tdms_file = TdmsFile.read(file_path)
    except Exception as e:
        print("Error reading TDMS file:", e)
        return

    groups = tdms_file.groups()
    if len(groups) == 0:
        raise ValueError("No groups found.")

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

    # ---- Build channel labels ----
    if sensor_type == "Current":
        def get_label(raw_name):
            return cfg["channel_labels"].get(raw_name, raw_name)
    else:
        num_channels = len(channels)
        if num_channels == 1:
            def get_label(raw_name):
                return "Vibration"
        else:
            label_map = {ch.name: f"Vibration {i+1}" for i, ch in enumerate(channels)}
            def get_label(raw_name):
                return label_map.get(raw_name, raw_name)

    # Read signals and check sampling rate
    signals = []
    channel_names = []
    sampling_rate = None
    duration = None
    n_samples_total = None

    for ch in channels:
        signal = ch[:]
        signals.append(signal)
        channel_names.append(ch.name)

        si = ch.properties.get("wf_increment")
        if si is None:
            raise KeyError(f"wf_increment missing for {ch.name}")
        current_sr = 1.0 / si

        if sampling_rate is None:
            sampling_rate = current_sr
            duration = len(signal) * si
            n_samples_total = len(signal)
        else:
            if abs(current_sr - sampling_rate) > 1e-6:
                raise ValueError(f"Sampling rate mismatch: {sampling_rate} vs {current_sr}")

    # ---- Compute FFT for each channel ----
    freq_list = []
    amp_list = []
    dominant_freqs = []
    dominant_amps = []
    for sig in signals:
        freq, amp = compute_fft(sig, sampling_rate)
        freq_list.append(freq)
        amp_list.append(amp)

        # Ignore DC component (index 0) when finding the dominant frequency
        idx_max = np.argmax(amp[1:]) + 1   # start from bin 1
        dom_freq = freq[idx_max]
        dom_amp = amp[idx_max]
        dominant_freqs.append(dom_freq)
        dominant_amps.append(dom_amp)

    # ---- Determine the maximum display frequency safely ----
    desired_max = cfg.get("max_display_freq", 5000)
    # Clamp to Nyquist frequency to avoid plotting beyond the valid range
    max_display_freq = min(desired_max, sampling_rate / 2)

    # ---- Console output ----
    print("\n" + "="*70)
    print(f"         FFT ANALYSIS – {sensor_type.upper()} SIGNAL")
    print("="*70)
    print(f"Motor Rating      : {motor_rating}")
    print(f"Dataset           : {dataset_family}")
    print(f"Fault Type        : {fault_type}")
    print(f"Fault Severity    : {fault_severity}")
    print("-"*70)
    print(f"Sampling Rate     : {sampling_rate:.0f} Hz")
    print(f"Number of Samples : {n_samples_total}")
    print(f"Duration          : {duration:.4f} s")
    # FFT bins
    n_fft_bins = len(freq_list[0])   # same for all channels
    print(f"FFT Bins          : {n_fft_bins}")
    print("-"*70)
    freq_res = sampling_rate / n_samples_total
    nyquist = sampling_rate / 2.0
    print(f"Frequency Resolution : {freq_res:.5f} Hz")
    print(f"Nyquist Frequency    : {nyquist:.1f} Hz")
    print(f"Display Range        : 0 – {max_display_freq:.0f} Hz")
    print("-"*70)
    print("Dominant Frequencies (excluding DC)")
    labels_printed = [get_label(name) for name in channel_names]
    for label, dom_f, dom_a in zip(labels_printed, dominant_freqs, dominant_amps):
        print(f"  {label:<12} : {dom_f:.2f} Hz  (amplitude = {dom_a:.6f})")
    print("="*70)
    print("\nNOTE: FFT computed on the entire signal. For feature extraction,\n"
          "windowing (e.g., 1024‑point) will be used in the preprocessing stage.\n")

    # ---- Save FFT plot ----
    labels_plot = [get_label(name) for name in channel_names]
    save_path = os.path.join(
        r"C:\FinalYearProject\Figures",
        cfg["save_folder"],
        "FFT",
        f"fft_combined_{motor_rating}_{dataset_family}_{fault_type}_{fault_severity}.png"
    )
    plot_fft(freq_list, amp_list, labels_plot, cfg,
             motor_rating, dataset_family, fault_type, fault_severity,
             sampling_rate, max_display_freq, save_path)

    # ---- Execution time ----
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds")
    print("="*70)


if __name__ == "__main__":
    main()