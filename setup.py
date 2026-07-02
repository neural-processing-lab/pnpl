from setuptools import find_packages, setup

setup(
    name="pnpl",
    version="0.2.0",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "pnpl.competition": ["data/*.csv"],
    },
    install_requires=[
        "mne",
        "mne_bids",
        "numpy",
        "pandas",
        "torch",
        "h5py",
        "huggingface_hub",
        "requests",
        # Kaggle submission upload. >= 2.0 supports the modern KAGGLE_API_TOKEN
        # (KGAT_…) format and requires Python >= 3.11; older Pythons resolve to
        # the 1.7.x line which falls back to KAGGLE_USERNAME + KAGGLE_KEY auth.
        "kaggle",
    ],
    author="Dulhan Jayalath",
    author_email="dulhan@robots.ox.ac.uk",
    description="Load and process brain datasets for deep learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/neural-processing-lab/pnpl",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
)
