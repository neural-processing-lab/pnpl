from setuptools import find_packages, setup

setup(
    name="pnpl",
    version="0.0.6",
    packages=find_packages(),
    install_requires=[
        "mne",
        "mne_bids",
        "numpy",
        "pandas",
        "torch",
        "h5py",
        "huggingface_hub"
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
