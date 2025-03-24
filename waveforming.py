import numpy as np
from scipy.io import wavfile
import scipy.signal as ss
import matplotlib.pyplot as plt
from conf import MICROPHONES, SOURCES


L = len(SOURCES)

# Read all microphone files
fs = None
signals = []
for m in MICROPHONES:
    current_fs, data = wavfile.read(m["filename"])
    if len(data.shape) > 1:
        data = data[:, 0]
    data = data.astype(np.float32)
    if fs is None:
        fs = current_fs
    else:
        assert fs == current_fs, "All files must have the same sample rate"
    signals.append(data)
signals = np.array(signals)

c = 343.0  # Speed of sound

angles = np.arange(0, 181, 1)
power = np.zeros(len(angles))

for i, theta in enumerate(angles):
    theta_rad = np.deg2rad(theta)
    delays = []
    for m in MICROPHONES:
        x, y = m["position"]
        # Corrected delay calculation
        delay = (x * np.cos(theta_rad) + y * np.sin(theta_rad)) / c
        delays.append(delay)
    sample_delays = np.array(delays) * fs
    sample_shifts = np.round(sample_delays).astype(int)

    shifted_signals = []
    for sig, shift in zip(signals, sample_shifts):
        if shift > 0:
            shifted = np.hstack((np.zeros(shift), sig[:-shift]))
        elif shift < 0:
            shifted = np.hstack((sig[-shift:], np.zeros(-shift)))
        else:
            shifted = sig
        shifted_signals.append(shifted)
    summed_signal = np.sum(shifted_signals, axis=0)
    power[i] = np.sum(summed_signal**2)

peaks, _ = ss.find_peaks(power, width=5)

# If we didn't find exactly L peaks, take the L highest peaks
if len(peaks) != L:
    # Sort peaks by height and take the L highest
    peak_heights = power[peaks]
    idx = np.argsort(peak_heights)[::-1]  # Sort in descending order
    peaks = peaks[idx[:L]]  # Take the L highest peaks

center_x = np.mean([m["position"][0] for m in MICROPHONES])
real_doas = []
for source in SOURCES:
    x, y = source["position"]
    # Corrected real DOA calculation
    angle_rad = np.arctan2(y, x - center_x)
    angle_deg = np.rad2deg(angle_rad)
    angle_deg = angle_deg % 360  # Normalize to 0-360
    if angle_deg > 180:
        angle_deg = 360 - angle_deg  # Fold into 0-180 range
    real_doas.append(angle_deg)
real_doas_sorted = np.sort(real_doas)

print(f"Estimated DOA angle: {angles[peaks]} degrees")
print(f"Real DOAs: {real_doas_sorted} degrees")

plt.figure(figsize=(10, 6))
plt.plot(angles, power)
plt.plot(angles[peaks], power[peaks], "x")
for ang in real_doas_sorted:
    plt.axvline(x=ang, color="r", linestyle="--", label=f"First Real DOA: {ang:.2f}°")
plt.xlabel("Angle (degrees)")
plt.ylabel("Power")
plt.title("DOA Estimation using Delay-and-Sum Beamforming")
plt.grid(True)
plt.show()
