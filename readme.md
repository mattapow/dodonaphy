# Dodonaphy

Hyperbolic embeddings for approximating phylogenetic posteriors.

This code is under active research & development and not intended for production use for phylogenetic inference.


## Installing
Create a pip environment.
Use python version 3.9 because:
a) on Mac OS Catalina, CommandLineTools (possibly used in setuptools) has an error in python==3.8.2.
b) Numpy does not yet support python>=3.10.
```
python3.9 -m pip venv env
source env/bin/activate
```
First, install the common requirements (because Numpy and Cython are needed before installing Dodonaphy)
```
python3 -m pip install -r requirements/common.txt
```

Then install the Dodonaphy package using pip:
```
pip install dodonaphy
```
Alternatively the package can be installed locally using
```
pip install -e .
```

## Running tests
Once the package is installed, tests are run using pytest:
```
pytest
```

## Model
The basic idea is to embed points in Hyperbolic space and connect them to form a tree.
Perform Bayesian inference (MCMC or Variational Inference) in the embedding space.
Gradient ascent is performed using pytorch.
See doc/notes.pdf for a description of the embedding models being used.

## Basic usage
Run Dodonaphy using
```
python src/run.py [OPTIONS]
```
At the command line, type "python src/run.py -h" to see the options.

## Post-processing
Several pre-processing and post-processing functions are in the post_process folder.
To install the additional packages used for this, run:
```
python3 -m pip install -r requirements/post_processing.txt
```
