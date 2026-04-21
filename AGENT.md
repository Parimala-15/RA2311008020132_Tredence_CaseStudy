# Agent Execution Instructions

## Context
This repository contains a PyTorch implementation of a custom neural network (`SelfPruningMLP`) applied to the CIFAR-10 dataset. The code demonstrates dynamic weight pruning using learnable gate scores.

## Environment Setup
To evaluate this code, please initialize a Python 3.10+ environment and install the following dependencies:
`pip install torch torchvision matplotlib numpy jupyter`

*Note: The code is hardware-agnostic and will automatically fallback to CPU execution if a CUDA-enabled GPU is not available.*

## Execution Steps
The primary executable is a Jupyter Notebook (`solution.ipynb`). 
To run the notebook automatically and verify the output, use the following command:
`jupyter nbconvert --to notebook --execute solution.ipynb`

## Expected Artifacts
Upon successful execution, the script will generate two visual reports in the working directory:
1. `gate_distribution.png`
2. `spatial_heatmap.png`

Please verify the presence of these files to confirm a successful run. Check `REPORT.md` for the theoretical explanation of these results.
