# scCausalVI

[![Documentation Status](https://readthedocs.org/projects/scvi-tools/badge/?version=stable)](https://sccausalvi.readthedocs.io/en/latest/)
[![PyPI Downloads](https://static.pepy.tech/badge/sccausalvi)](https://pepy.tech/projects/sccausalvi)
[![PyPI Downloads](https://static.pepy.tech/badge/sccausalvi/month)](https://pepy.tech/projects/sccausalvi)

scCausalVI is a causality-aware generative model designed to disentangle inherent cellular heterogeneity from treatment effects in single-cell RNA sequencing data, particularly in case-control studies.

![scCausalVI Overview](https://github.com/ShaokunAn/scCausalVI/blob/main/sketch/method%20overview.png)


## Introduction

scCausalVI addresses a major analytical challenge in single-cell RNA sequencing: distinguishing inherent cellular variation from extrinsic cell-state-specific effects induced by external stimuli. The model:

- Decouples intrinsic cellular states from treatment effects through a deep structural causal network
- Explicitly models causal mechanisms governing cell-state-specific responses
- Enables cross-condition in silico prediction
- Accounts for technical variations in multi-source data integration
- Identifies treatment-responsive populations and molecular signatures

### Key Features of scCausalVI
- Interpretable and disentangled latent representation
- Data integration
- In silico perturbation
- Identification of treatment-responsive populations

## Installation
There are several alternative options to install scCausalVI:

1. Install the latest version of scCausalVI via pip:

   ```bash
   pip install scCausalVI
   ```

2. Or install the development version via pip:

   ```bash
   pip install git+https://github.com/ShaokunAn/scCausalVI.git
   ```

## Examples
See examples at our [documentation site](https://sccausalvi.readthedocs.io/en/latest/tutorial.html).

## Reproducing Results

In order to reproduce paper results visit [here](https://github.com/ShaokunAn/scCausalVI-reproducibility/tree/main).


## References

For a detailed explanation of our methods, please refer to our [cell systems](https://www.cell.com/cell-systems/abstract/S2405-4712(25)00276-5) manuscript.

## Contact

Feel free to contact us by [mail](shan12@bwh.harvard.edu). If you find a bug, please use the [issue tracker](https://github.com/ShaokunAn/scCausalVI/issues).
