# Clustering Nanopore Current Traces with Sparse & Deep Learning Methods

This project investigates **unsupervised clustering of biological nanopore current signals** to distinguish peptides and their post-translational modifications (PTMs).

## Overview

Nanopore sensing produces complex current signals where traditional features (e.g., mean, dwell time) are insufficient. This work explores **richer feature representations and advanced clustering methods** to improve analyte differentiation. 

## Approach

* **Feature Extraction**

  * TSFEL, catch22, and domain-specific nanopore features
  * Removal of mean-dependent features to focus on dynamics

* **Clustering Methods**

  * Sparse K-Means with L1-based feature weighting
  * Triplet-based deep clustering for latent space separation
  * Memory-Augmented Attention Triplet Network (continual learning + feature attention) 

* **Feature Analysis**

  * Elastic Net with bootstrap-based Variable Inclusion Frequency (VIF)
  * Identification of features contributing to latent representations 

## Results

* Strong clustering performance across analytes (**ARI up to ~0.947**) 
* Clear separation between **baseline vs blockade signals**
* Sparse feature weighting improves interpretability and performance
* Deep metric learning captures subtle differences between PTMs

## Key Insights

* Mean-based features alone are insufficient for nanopore data
* Feature selection is critical in high-dimensional settings
* Deep clustering significantly improves fine-grained discrimination
* Attention + memory mechanisms help in sequential/continual learning

## Future Work

* More robust hyperparameter selection (e.g., sparsity tuning)
* Improved triplet mining strategies
* Scaling to larger and more diverse biological datasets
