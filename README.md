# Self-Pruning MLP on CIFAR-10

This repository contains the code and documentation for a custom PyTorch neural network that dynamically prunes its own weights during training. 

## Project Overview
The core of this project is the `PrunableLinear` layer, which utilizes learnable gate scores and a custom sparsity loss function (L1 penalty) to induce network sparsity without sacrificing accuracy on the CIFAR-10 dataset.

## Repository Structure
* `solution.ipynb`: The main executable notebook containing the model architecture, training loop, and evaluation metrics.
* `REPORT.md`: A detailed breakdown of the methodology, loss function mechanics, and analysis of the resulting sparsity.
* `AGENT.md`: Instructions for automated LLM agents or CI/CD pipelines to execute the code.

## Quick Start
To run the code locally, ensure you have the necessary dependencies installed:
```bash
pip install torch torchvision matplotlib numpy jupyter
