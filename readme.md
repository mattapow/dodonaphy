# Dodonaphy

Hyperbolic embeddings for approximating phylogenetic posteriors.

This code is under active research & development and not intended for production use for phylogenetic inference.

## Installing with Conda and Pip
First install the Python version of hydra+ from https://github.com/mattapow/hydraPlus.

On Mac OS Catalina, CommandLineTools (possibly used in setuptools) has an error in python==3.8.2. So use python 3.9 or 3.10.

```
conda create --name dodonaphy python=3.9
conda activate dodonaphy
```

Pip needs numpy and cython for the setup:

```
pip install cython
pip install numpy
```

Then install the Dodonaphy locally package using pip:
```
pip install -e .
```


## Running tests
(Optional, requires pytest) Once the package is installed, tests are run using pytest:
```
pytest -o "testpaths=test"
```

## Model
The basic idea is to embed genetic data as points in Hyperbolic space and connect them to form a tree.
Then perform Bayesian inference (MCMC or Variational Inference) in the embedding space.
Gradient ascent is performed using pytorch.

## Basic usage
Run Dodonaphy using
```
dodo [OPTIONS]
```
As an example, data in the example directory is provided.
Perform MCMC with a gamma dirichlet prior on the example data as follows.
```
dodo --infer mcmc --path_root example --prior gammadir --epochs 10000
```
This starts from the neighbour joining tree.
To start from a better tree, use:
```
dodo --infer mcmc --path_root example --prior gammadir --epochs 10000 --start start.nex --suffix good_start
```
This creates a directory contained in ```mcmc/up_nj/d3_k0```, signifying the embedding method "up", the tree connection method "nj" Neighbour Joining, three dimensions and a log negative curvature of 0, i.e. a curvature of -1.
This directory has:
- ```samples.t``` containing sampled trees in the posterior
- ```locations.csv``` containing the tip embedding locations
- ```mcmc.log``` with information about the MCMC invoked.



## Help
At the command line, type 
```dodo --help```
to see the options.
