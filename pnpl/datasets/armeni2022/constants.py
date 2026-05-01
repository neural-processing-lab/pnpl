"""
Constants for the Armeni et al. (2022) dataset.

Source: Radboud Data Repository (DSC_3011085.05_995_v1).
Three participants listening to ~10 hours of audiobook ("compr"
comprehension task) recorded with a CTF MEG system.
"""

RADBOUD_DATASET_URL = "https://webdav.data.ru.nl/dccn/DSC_3011085.05_995_v1/"

SUBJECTS = ["001", "002", "003"]
SESSIONS = [f"{i:03d}" for i in range(1, 11)]
TASKS = ["compr"]
