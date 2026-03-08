import sys
from pathlib import Path
# 1. Get the absolute path of the directory containing 'ScalogramGeneration'
# .parent refers to ScalogramGeneration, .parent.parent refers to the ProjectRoot
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
import src.scalogram_cnn_project.settings.config as config


import mne
import matplotlib.pyplot as plt

ITA_PILOT_DIR = config.ITA_PILOT_DIR

# preload=True loads the data into memory.
# edf_file = mne.io.read_raw_edf(ITA_PILOT_DIR / "18062024voo1.EDF", preload=True)

eye_open   = mne.io.read_raw_edf(ITA_PILOT_DIR / "basal18062024OA.EDF", preload=True)

eye_closed = mne.io.read_raw_edf(ITA_PILOT_DIR / "basal18062024OF.EDF", preload=True)



print(eye_open.ch_names)

print(eye_closed.ch_names)


eye_open.plot(block=False, title="Eye Open Raw")
eye_closed.plot(block=False, title="Eye Closed Raw")
channels_of_interest = ['O1', 'O2']



spectrum_open = eye_open.compute_psd(fmin=0, fmax=30, tmin=10, tmax=40, picks=channels_of_interest)
fig_open = spectrum_open.plot(show=False)
fig_open.suptitle("Eye Open PSD")

spectrum_closed = eye_closed.compute_psd(fmin=0, fmax=30, tmin=10, tmax=40, picks=channels_of_interest)
fig_closed = spectrum_closed.plot(show=False)
fig_closed.suptitle("Eye Closed PSD")


plt.show()