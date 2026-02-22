from setuptools import setup, find_packages

setup(
    name="masd",
    version="1.0.0",
    author="Taiba Majid Wani, Irene Amerini",
    author_email="majid@diag.uniroma1.it",
    description="MASD: Multi-Scale Artifact-Aware Self-Supervised Deepfake Detector",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/TaibaMajidWani/MASD",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "torchaudio>=0.12.0",
        "numpy>=1.21.0",
        "librosa>=0.9.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "pyyaml>=6.0",
    ],
)
