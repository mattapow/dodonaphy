# Dodonaphy

Hyperbolic embeddings for approximating phylogenetic posteriors.

This code is under active research & development and not intended for production use for phylogenetic inference.

## Installing with Conda and Pip
First install the Python version of hydra+ from https://github.com/mattapow/hydraPlus.

On Mac OS Catalina, CommandLineTools (possibly used in setuptools) has an error in python==3.8.2. Numpy does not yet support python>=3.10. So use 3.9

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
Once the package is installed, tests are run using pytest:
```
pytest -o "testpaths=test"
```

## Model
The basic idea is to embed points in Hyperbolic space and connect them to form a tree.
Perform Bayesian inference (MCMC or Variational Inference) in the embedding space.
Gradient ascent is performed using pytorch.
See doc/notes.pdf for a description of the embedding models being used.

## Basic usage
Run Dodonaphy using
```
dodo [OPTIONS]
```

## Help
At the command line, type 
```dodo --help```
to see the options.
