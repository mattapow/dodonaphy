#!/usr/bin/env Rscript

library(ape)
library(hydra)
library(phangorn)
library(rstan)
library(cmdstanr)

# NOTE: it is necessary to set the path to cmdstan. uncomment the below
# set the path appropriately for your system
# set_cmdstan_path("~/software/cmdstan-2.21.0")

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

# also create an embedding with internal nodes
emm <- hydraPlus(acosh(exp(dist.nodes(simtree))), dim = dim, curvature = 1)

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
#dphy_dat <- list(D=dim, L=seqlen, S=nseqs, tipdata=tipdata, leaf_r=hpembedding$r, leaf_dir=hpembedding$directional)
dphy_dat <- list(D=dim, L=seqlen, S=nseqs, tipdata=tipdata, leaf_r=emm$r[1:6], leaf_dir=emm$directional[1:6,])
initint<-list(list(int_r=emm$r[8:11],int_dir=emm$directional[8:11,]))

file.copy("src/dodonaphy_cpp_utils.hpp", paste(tempdir(),"/user_header.hpp", sep=""), overwrite=TRUE)
dphy_mod <- cmdstan_model("src/dodonaphy.stan",stanc_options=list(allow_undefined=TRUE))
vbfit <- dphy_mod$variational(data=dphy_dat,tol_rel_obj=0.001,output_samples=1000)

blens<-vbfit$draws("blens")
peels<-vbfit$draws("peel")



# extracts a newick format string that represents the tree
tree_to_newick<-function(tip_names, peel_row, blen_row){
	newick <- ""
	chunks <- list()
	plen <- length(tip_names)-1
	for(p in 1:plen){
		n1 <- peel_row[p]
		n2 <- peel_row[p+plen]
		n3 <- peel_row[p+2*plen]
		if(n1 <= length(tip_names)){
			chunks[n1] <- paste(tip_names[n1], ":", blen_row[n1], sep="")
		}
		if(n2 <= length(tip_names)){
			chunks[n2] <- paste(tip_names[n2], ":", blen_row[n2], sep="")
		}
		n3name <- paste(as.character(n3), ":", blen_row[n3], sep="")
		if(p == plen){
			n3name <- ";"
		}
		chunks[n3] <- paste("(", chunks[n1], ",", chunks[n2], ")", n3name, sep="")
	}
	as.character(chunks[peel_row[length(peel_row)]])
}

testtree<-tree_to_newick(names(dnaseq), peels[30,], blens[30,])
ttt1<-read.tree(text=testtree)
plot(ttt1)
add.scale.bar()



