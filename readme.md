# Dodonaphy

Hyperbolic embeddings for approximating phylogenetic posteriors.

This code is under active research & development and not intended for production use for phylogenetic inference.


## Installing
Install the dodonaphy package using pip:
```
pip install dodonaphy
```
alternatively the package can be installed locally using
```
pip install -e .
```

## Running tests
Once the package is installed tests are run using pytest:
```
pytest
```

## Model
The basic idea is embed points in Hyperbolic space and connect them to form a tree.
Perform Bayesian inference (MCMC or Variational Inference) in the embedding space.
Gradient ascent is performed using pytorch.
See doc/notes.pdf for a description of the embedding models being used.

## Basic usage
Run dodonaphy using
```
dodonaphy [dim]
```
where dim is the embedding dimension e.g. "dodonaphy 4".
The entry point is run.py in the src folder and at the moment, most options must be specified as variables in run.py.

## Post-processing
A number of pre- and post-processing functions are in the post_process folder.