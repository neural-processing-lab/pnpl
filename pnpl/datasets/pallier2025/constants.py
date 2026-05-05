"""
Constants for the Pallier et al. (2025) "LittlePrince — Listen" MEG dataset.

Source: OpenNeuro `ds007523 <https://openneuro.org/datasets/ds007523>`_
(LittlePrince_MEG_French_Listen_Pallier2025).

58 native French adults passively listening to the full French audiobook
of "Le Petit Prince" (Antoine de Saint-Exupéry) split into 9 segments,
recorded with an Elekta Neuromag TRIUX MEG system (306 channels:
102 magnetometers + 204 planar gradiometers, 1000 Hz, online HP 0.1 Hz /
LP 330 Hz, line frequency 50 Hz).
"""

OPENNEURO_DATASET_ID = "ds007523"
OPENNEURO_SNAPSHOT_TAG = "1.0.1"

SUBJECTS = [f"{i:02d}" for i in range(1, 59)]
SESSIONS = ["01"]
TASKS = ["listen"]
RUNS = [f"{i:02d}" for i in range(1, 10)]
