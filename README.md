# ECLipsE: Compositional-Estimation-of-Lipschitz-Constants
This repository is the codes for paper "ECLipsE: Efficient Compositional Lipschitz Constant Estimation for Deep Neural Networks" ([NeurIPS 2024 paper](https://proceedings.neurips.cc/paper_files/paper/2024/file/1419d8554191a65ea4f2d8e1057973e4-Paper-Conference.pdf)), providing efficient and tight Lipschitz Estimate for feedforword neural networks (FNNs). ECLipsE.m and ECLipsE_Fast.m are functions for the two algorithms proposed and we can obtain the estimates for the Lipschitz constants (strict and tight upper bound), time used, and the trivial Lipschitz upper bound by directly calling the functions with weights of FNNs. The weights should be in the format of W1,W2,..., etc. and the biases are not needed for our methods. The activation functions for the FNNs need to be slope-restricted in [0,1], which is satisfied by most activation functions.

## Install from PyPI

```bash
pip install eclipse-nn
```

## Publish a new PyPI version

1. Update the version in `pyproject.toml`.
2. Build distributions:

```bash
python -m build
```

3. Upload to PyPI:

```bash
python -m twine upload dist/*
```

4. Create a matching GitHub release/tag.

To reproduce the experiments in paper, we first generate / train FNNs as
1. For randomly generated neural networks, run generate_random_weights.py with corresponding depths and widths for FNNs. The weights of FNNs will be stored under datasets/random.
2. For neural networks trained on mnist dataset, run training_MNIST.py. The weights of FNNs will be stored under datasets/MNIST.
   
All the results can be obtained through Lip_estimates.m.


# ECLipsE-Gen-Local: Efficient Compositional Local Lipschitz Estimates for Deep Neural Networks
The code under folder ECLipsE_Gen_Local_matlab contains the code for algorithms series ECLipsE-Gen-Local. The algorithms give efficient, scalable and tight local Lipschitz estimates for deep FNN. Paper is accepted to TMLR in 2026.03, and is available at https://arxiv.org/abs/2510.05261.



# Installation
We also provide implementation in python under the "python" folder for all the algorithms. Users can directly install the package [![PyPI version](https://img.shields.io/pypi/v/eclipse-nn.svg)](https://pypi.org/project/eclipse-nn/) through `pip install eclipse-nn`.
