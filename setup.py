from setuptools import setup
from importlib.machinery import SourceFileLoader

with open("README.md") as file:
    long_description = file.read()

version = SourceFileLoader("deepafx_st.version", "deepafx_st/version.py").load_module()

setup(
    name="deepafx-st",
    version=version.version,
    description="DeepAFx-ST",
    author="See paper",
    author_email="See paper",
    url="https://github.com/adobe-research/DeepAFx-ST",
    packages=["deepafx_st"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Copyright Adobe Inc.",
    install_requires=[
        "torch==1.9.0",
        "torchaudio==0.9.0",
        "torchmetrics>=0.4.1",
        "torchvision==0.10.0",
        "audioread>=2.1.9",
        "auraloss>=0.2.1",
        "librosa>=0.8.1",
        "matplotlib",
        "numpy",
        "pytorch-lightning>=1.4.0",
        "SoundFile>=0.10.3.post1",
        "sox>=1.4.1",
        "tensorboard>=2.4.1",
        "scikit-learn>=0.24.2",
        "scipy",
        "pyloudnorm>=0.1.0",
        "julius>=0.2.6",
        "torchopenl3",
        "cdpam",
        "wget",
        "pesq",
        "umap-learn",
        "setuptools==58.2.0"
    ],
)
