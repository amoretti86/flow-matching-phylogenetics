# Introduction
This repository contains code for the paper
"Variational Combinatorial Sequential Monte Carlo for Bayesian Phylogenetics in Hyperbolic Space"
by Alex Chen, Philippe Chlenski, Kenneth Munyuza, Antonio Khalil Moretti, Christian A. Naesseth, and Itsik Pe'er.
The paper has been [accepted to AISTATS 2025](https://openreview.net/forum?id=29cqyUl2bI).

If you use this repository, please cite the paper:
```
@inproceedings{
chen2025variational,
title={Variational Combinatorial Sequential Monte Carlo for Bayesian Phylogenetics in Hyperbolic Space},
author={Alex Chen and Philippe Chlenski and Kenneth Munyuza and Antonio Khalil Moretti and Christian A. Naesseth and Itsik Pe'er},
booktitle={The 28th International Conference on Artificial Intelligence and Statistics},
year={2025},
url={https://openreview.net/forum?id=29cqyUl2bI}
}
```

# Setup
First, create a Python virtual environment:
```bash
python3 -m venv hvcsmc
source hvcsmc/bin/activate
```

Then, install the required packages:
```bash
pip install -e .
```

Next, create a [Wandb](https://wandb.ai) account to save run metrics. Link your
account to the CLI by running:
```bash
wandb login
```

# Example Runs
Notes:
- `--q-matrix`: specifies the Q matrix to use.
  - `jc69`: fixed JC69 Q matrix.
  - `stationary`: one global Q matrix with each entry free.
  - `mlp_factorized`: An MLP maps embeddings to holding times and stationary
    probabilities, forming the Q matrix. More memory efficient than `mlp_dense`.
  - `mlp_dense`: An MLP maps embeddings to all entries of the Q matrix.
- `--lookahead-merge`: performs H-VNCSMC if `--hyperbolic` is set, or VNCSMC
  otherwise.
- `--hash-trick`: memoizes compute over tree topologies to speed up computation.
  Only applies when `--hyperbolic` is set, and essentially required if
  `--lookahead-merge` is set.
- `--checkpoint-grads`: use gradient checkpointing to reduce memory usage.

Run H-VCSMC on primates using K=512 and a learned Q matrix:
```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 512 --q-matrix mlp_dense --hash-trick data/primates.phy
```

Run H-VNCSMC (nested proposal) on primates using K=16 and a learned Q matrix:
```bash
python -m scripts.train.hyp_train --lr 0.01 --epochs 200 --k 16 --q-matrix mlp_dense --lookahead-merge --hash-trick data/primates.phy
```

Run H-VNCSMC on a larger benchmark dataset (DS1) using K=16 and a factorized Q
matrix:
```bash
python -m scripts.train.hyp_train --lr 0.05 --epochs 200 --k 16 --q-matrix mlp_factorized --lookahead-merge --hash-trick data/hohna/DS1.phy
```

Run H-VNCSMC on benchmark datasets DS1-DS7, with deferred branch sampling to
better learn the embeddings:
```bash
python -m scripts.benchmarks.hyp_smc_benchmark --q-matrix mlp_factorized
```

# Extracting Data
After training a model, you can extract the nucleotide sequence distributions and embeddings in the Poincaré disk:

## Setup for Extraction
First, create an `extraction_utils.py` file in your project directory with the utility functions needed for extraction. The file should contain functions for extracting sequence distributions and embeddings, visualizing the Poincaré disk, and saving the extracted data.

Then create an extraction script called `extract_data.py` in the scripts directory.

## Running the Extraction
After your training process has completed, run the extraction script:

```bash
python -m scripts.extract_data --checkpoint-dir checkpoints --output-dir extracted_data
```

To use the checkpoint with the best performance instead of the most recent one:

```bash
python -m scripts.extract_data --checkpoint-dir checkpoints --output-dir extracted_data --best
```

## What the Extraction Provides
The extraction script will:
1. Load your trained model
2. Extract the sequence distributions (probabilities of A,C,G,T for each site in each species)
3. Extract the embeddings in the Poincaré disk (for all taxa and internal nodes)
4. Generate a visualization of the embeddings
5. Save all data in both PyTorch (.pt) and NumPy (.npy) formats to the specified output directory

# Visualizing the Phylogenetic Tree in the Poincaré Disk
To visualize the phylogenetic tree with proper geodesic edges in the Poincaré disk:

## Setup for Visualization
Create the following two files in your project directory:

1. `poincare_tree_visualization.py`: Contains the core functions for drawing the tree
2. `visualize_tree.py`: Script to load data and generate the visualization

## Running the Visualization
After extracting the data, run the visualization script:

```bash
python visualize_tree.py --embeddings-file extracted_data/all_embeddings.npy --results-dir extracted_data
```

To specify custom taxa names:
```bash
python visualize_tree.py --embeddings-file extracted_data/all_embeddings.npy --taxa-file taxa_names.txt
```

## Visualization Features
The tree visualization includes:
- Points for taxa (blue) and internal nodes (red)
- Labels for all nodes
- Geodesic edges showing the tree topology
- Proper hyperbolic geometry with all edges as geodesics in the Poincaré disk

This visualization accurately represents the hierarchical structure learned by the model in hyperbolic space.