#!/usr/bin/env Rscript

library(ape)
library(hydra)
library(phangorn)
library(rstan)
library(cmdstanr)

dim <- 3    # number of dimensions for embedding
nseqs <- 6  # number of sequences to simulate
seqlen <- 1000  # length of sequences to simulate

# function to give random branch lengths
shortb <- function(n) {
	runif(n, 0, 0.1)
}

# simulate a tree
simtree <- rtree(n=nseqs, rooted=TRUE, br=shortb)

# simulate a sequence alignment
dnaseq <- simSeq(simtree, l = seqlen)

#
# prepare data for dodonaphy stan
#

# compute an embedding of the tips
# get pairwise distances among leaf nodes
dists <- cophenetic.phylo(simtree)
# apply the transform suggested by Matsumoto et al 2020
tdists <- acosh(exp(dists))
# embed with hydraPlus
hpembedding <- hydraPlus(tdists, dim = dim, curvature = 1)

# glue the r together with the angles
leaf_locs <- cbind(hpembedding$r, hpembedding$directional)

# encodes sequences in a tip partial probability matrix
encode_tipdata <- function(dnaseq){
	tipdata<-array(data=0, dim=c(nseqs,seqlen,4))
	for(i in 1:seqlen){
		seqcol <- as.numeric(dnaseq[,i])
		for(j in 1:nseqs){
			tipdata[j,i,seqcol[j]]<-1
		}
	}
	tipdata
}
tipdata <- encode_tipdata(dnaseq)
dphy_dat <- list(D=dim, L=seqlen, S=nseqs, tipdata=tipdata, leaf_locs=leaf_locs)


file.copy("src/dodonaphy_cpp_utils.hpp", paste(tempdir(),"/user_header.hpp", sep=""), overwrite=TRUE)
dphy_mod <- cmdstan_model("src/dodonaphy.stan",stanc_options=list(allow_undefined=TRUE))
dphy_mod$variational(data=dphy_dat)


