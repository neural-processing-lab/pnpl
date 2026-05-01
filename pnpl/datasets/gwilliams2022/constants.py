"""
Constants for the MEG-MASC dataset (Gwilliams et al., 2022).

OSF project: https://osf.io/ag3kj/  (paper: https://arxiv.org/abs/2208.11488)

The dataset is split across four sibling OSF components because of OSF's
per-component storage cap; the OSFDownloadMixin transparently aggregates
files across all four.
"""

# OSF nodes that together hold the dataset's osfstorage. The primary node
# (ag3kj) holds the dataset_description.json / participants.tsv; the others
# split the per-subject MEG data.
OSF_PROJECT_ID = "ag3kj"
OSF_PROJECT_FALLBACKS = ["h2tzn", "u5327", "dr4wy"]

# Subjects, sessions, and tasks declared by the published BIDS dataset.
# Some subjects in the public release only have data for one session; the
# dataset class skips run keys whose files are not in the OSF manifest.
SUBJECTS = [f"{i:02d}" for i in range(1, 28)]
SESSIONS = ["0", "1"]
TASKS = ["0", "1", "2", "3"]

# Story title for each task index (from the project README).
TASK_STORIES = {
    "0": "lw1",
    "1": "cable_spool_fort",
    "2": "easy_money",
    "3": "The_Black_Widow",
}
