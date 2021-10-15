# Dodonaphy

Hyperbolic embeddings for approximating phylogenetic posteriors.

This code is under active research & development and not intended for production use for phylogenetic inference.


## Installing
Create a pip envirnment.
On Mac OS Catalina, CommandLineTools (possibly used in setuptools) has error in python==3.8.2.
But numpy isn't yet supported in python>=3.10.
```
python3.9 -m pip venv env
source env/bin/activate
```
First install the common requirements (because numpy and cython are needed before installing dodonaphy)
```
python3 -m pip install -r requirements/common.txt
```

Then install the dodonaphy package using pip:
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
To install the additional packages used here run:
```
python3 -m pip install -r requirements/post_processing.txt
```