from setuptools import setup, find_packages

setup(
    name="scCausalVI",
    version='0.0.10',
    author="Shaokun An",
    author_email="shan12@bwh.harvard.edu",
    description="A deep causality-aware model for disentangling treatment effects at single-cell resolution "
                "for perturbational scRNA-seq data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ShaokunAn/scCausalVI/",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "scanpy>=1.9.6",
        "torch>=2.0.0",
        "anndata>=0.10.3",
        "numpy>=1.23.5,<2.0",
        "setuptools>=59.5.0",
        "pandas>=2.1.1",
        "matplotlib>=3.8.1",
        "scikit-learn>=1.3.2",
        "tqdm>=4.66.1",
        "seaborn>=0.12.2",
        "scipy>=1.11.3",
        "scvi-tools>=0.16.1",
        "pytorch-lightning>=1.5.10",
        "gdown>=5.2.0"
    ],
    python_requires=">=3.9",
)
