from nptdms import TdmsFile
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# STEP 1 : TDMS File Path
# =====================================================
file_path = r"C:\FinalYearProject\Dataset\1.0kW\current\1000W_0_00_current_coil.tdms"

# =====================================================
# STEP 2 : Read TDMS File
# =====================================================
tdms_file = TdmsFile.read(file_path)

# =====================================================
# STEP 3 : Read ai0 Channel
# =====================================================
channel = tdms_file["Log"]["cDAQ1Mod2/ai0"]

# =====================================================
# STEP 4 : Convert to NumPy Array
# =====================================================
current_signal = channel[:]

# =====================================================
# STEP 5 : Read Sampling Interval
# =====================================================
sampling_interval = channel.properties["wf_increment"]

# =====================================================
# STEP 6 : Create Time Axis
# =====================================================
time = np.arange(len(current_signal)) * sampling_interval

# =====================================================
# STEP 7 : Print Information
# =====================================================
print("=" * 60)
print("CHANNEL INFORMATION")
print("=" * 60)

print(f"Channel Name      : {channel.name}")
print(f"Number of Samples : {len(current_signal)}")
print(f"Shape             : {current_signal.shape}")
print(f"Data Type         : {current_signal.dtype}")

print("\nSIGNAL STATISTICS")
print("-" * 60)

print(f"Minimum Value : {np.min(current_signal)} A")
print(f"Maximum Value : {np.max(current_signal)} A")
print(f"Mean Value    : {np.mean(current_signal)} A")

print("\nSampling Interval :", sampling_interval, "seconds")
print("Sampling Frequency:", 1/sampling_interval, "Hz")

print("\nFIRST 10 CURRENT SAMPLES")
print("-" * 60)
print(current_signal[:10])

# =====================================================
# STEP 8 : Plot using Time
# =====================================================
plt.figure(figsize=(12,5))

plt.plot(time[:5000], current_signal[:5000])

plt.title("Healthy PMSM Current Signal (Time Domain)")
plt.xlabel("Time (seconds)")
plt.ylabel("Current (A)")

plt.grid(True)

plt.savefig(
    r"C:\FinalYearProject\Figures\Healthy\Healthy_Current_Time_Domain.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nFigure saved successfully.")
print("=" * 60)