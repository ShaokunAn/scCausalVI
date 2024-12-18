scCausalVI
======================================
scCausalVI is a causality-aware model for analyzing single-cell RNA sequencing
data, designed to disentangle treatment effects from intrinsic cellular heterogeneity.
Using a deep generative framework, scCausalVI enables precise, single-cell-level inference
of gene expression responses to experimental treatments by modeling the causal dependencies
between treatment effects and cellular states. Key features include support for clustering,
treatment response analysis, and counterfactual inference, allowing users to explore cellular
variability and identify treatment-responsive subpopulations.

.. image:: _static/overview.png
   :alt: Overview


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   self
   installation
   tutorial
   api

.. include:: README.md
   :parser: myst_parser.sphinx_
