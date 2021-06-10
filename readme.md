# Dodonaphy Experiments

Scripts in src are for running experiments with dodonaphy package.

## Installing

First install dodonaphy, see ./dodonaphy/readme.md.

## Running

In src, invoke the run.py to run the simulations:

    >> python3 run.py

## Comparing with BEAST or MrBayes
Each run will generate a dna.nex file that was used as input for the analysis.
Run an MCMC (BEAST or MrBayes) on this file for comparison.

## Install asdsf for post processing

For post processing the asdsf, install bali-phy https://github.com/bredelings/BAli-Phy
and copy the trees-bootstrap and compare-runs.R files into ext. Using the compare-run.R
file requires having R installed.

## Post processing

Post process them using locations_kde.py and stat_cmp.py.

To process the asdsf using bali-phy files in ext, first run trees-boostrap as

    >> trees-bootstrap dir-1/C1.trees dir-2/C1.trees ... dir-n/C1.trees --LOD-table=LOD-table > partitions.bs 

Then plot the splits using

    >> R --slave --vanilla --args LOD-table compare-SF.pdf < compare-runs.R
