import numpy as np
import matplotlib.pyplot as plt
import pywt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# ==============================
# 1. Generate signal
# ==============================

## Size of the first scalogram generated, according to A. Zayed (2025)
width_px = 662
height_px = 536
dpi = 100

## Frequencies represented in the scalogram
freq_min = 1
freq_max = 100

sfreq = 500  # sampling frequency (Hz)
T = 30     # duration (seconds)
t = np.linspace(0, T, sfreq*T, endpoint=False)

signal = np.cos(2 * np.pi * 10 * t)  # 10 Hz cosine

# ==============================
# 2. Continuous Wavelet Transform
# ==============================

wavelet_type = 'cmor2.0-3.0'


freqs = np.linspace(freq_min, freq_max, 256)
scales = pywt.frequency2scale(wavelet_type, freqs * 1/sfreq)


coef, _ = pywt.cwt(signal, scales, wavelet_type, sampling_period=1/sfreq)
power = np.abs(coef)**2

# ==============================
# 3. Plot TFR (Scalogram)
# ==============================

# Adjusting the color contrast in the graph
vmin = np.percentile(power, 25)
vmax = vmin+20


fig = plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
ax = fig.add_axes([0.15, 0.15, 0.7, 0.75])


time = np.linspace(0, T, power.shape[1])
ax.set_title(f"Fz ({freq_min}-{freq_max} Hz)")
ax.set_ylabel("Frequency (Hz)")
ax.set_xlabel("Time (s)")
pcm = ax.pcolormesh(
    time,
    freqs,
    power,
    shading='auto',
    cmap='viridis',
    vmin=vmin,
    vmax=vmax
)


cax = fig.add_axes([0.92, 0.1, 0.02, 0.8])
fig.colorbar(pcm, cax=cax, label="Power")


folder_name = Path("CosineTFRExample")
fig_name = f'cosine_wavelet_{wavelet_type}.png'


fig.savefig(
    fname = folder_name / fig_name,
    dpi=dpi
)