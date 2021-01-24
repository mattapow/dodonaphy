functions {
// computes the euclidean distance between a leaf and an internal node
real distance(int node_1, int node_2, int D, real[,] locs_1, real[,] locs_2) {
    real dist = 0;
    for(i in 1:D){
        dist += square(locs_1[node_1,i] - locs_2[node_2,i]);
    }
    return sqrt(dist)+0.000000000001; // add a tiny amount to avoid zero-length branches
}


// computes the hyperbolic distance of two points given in radial/directional coordinates in the Poincare ball
// code taken from hydra R package and translated to Stan
real hyperbolic_distance(real r1, real r2, real[] directional1, real[] directional2, real curvature) {
	// force between numerical -1.0 and 1.0 to eliminate rounding errors
	real dp1[2];
	real iprod;
	real acosharg;
	real dist;
	dp1[1] =  dot_product(directional1,directional2);
	dp1[2] = -1.0;
	dp1[1] = max(dp1);
	dp1[2] = 1.0;
	iprod = min(dp1);
//	real iprod = min(max(-1.0,dot_product(directional1,directional2)),1.0);
	// hyperbolic 'angle'; force numerical >= 1.0
	dp1[1] = 2*(r1^2 + r2^2 - 2*r1*r2*iprod)/((1 - r1^2)*(1 - r2^2));
	dp1[2] = 0.0;
	acosharg = 1.0 + max(dp1);
	// hyperbolic distance between points i and j
	dist = 1/sqrt(curvature) * acosh(acosharg);
	return dist+0.000000000001; // add a tiny amount to avoid zero-length branches
}

void make_peel(real[,] leaf_locs, real[,] int_locs, int[,] peel, int[] location_map);

}


data {
    int D; // embedding dimension

    int <lower=0> L;               // alignment length
    int <lower=0> S;               // number of tip sequences
    real<lower=0,upper=1> tipdata[S,L,4];

    real leaf_locs[S,D]; // position of each tip sequence in the embedding
}

transformed data {
    int bcount; // number of branches
    bcount = 2*S-2;
}

parameters {
    real int_locs[S-3,D]; // unknown positions of internal nodes
}

model {

    // compute the phylogenetic likelihood
    // first generate a tree
    // then compute a likelihood

	vector[4] partials[2*S,L];   // partial probabilities for the S tips and S-1 internal nodes
	matrix[4,4] fttm[bcount]; // finite-time transition matrices for each branch
	vector[bcount] blens; // branch lengths
	int peel[S-1,3];   // list of nodes for peeling. this gets initialized via an additional c++ function 
	int location_map[2*S];   // list of nodes for peeling. this gets initialized via an additional c++ function 
    
	make_peel(leaf_locs, int_locs, peel, location_map);

	for( b in 1:(S-1) ){
		if(peel[b,1] <= S){
			// leaf to internal
			blens[peel[b,1]] = distance(peel[b,1], location_map[peel[b,3]]-S, D, leaf_locs, int_locs);
		}else{
			blens[peel[b,1]] = distance(location_map[peel[b,1]], location_map[peel[b,3]]-S, D, int_locs, int_locs);
		}
		if(peel[b,2] <= S){
			// leaf to internal
			blens[peel[b,2]] = distance(peel[b,2], location_map[peel[b,3]]-S, D, leaf_locs, int_locs);
		}else{
			blens[peel[b,2]] = distance(location_map[peel[b,2]], location_map[peel[b,3]]-S, D, int_locs, int_locs);
		}
	}

    
	for( b in 1:bcount ) {
	    for( i in 1:4 ) {
        	for( j in 1:4 ) {
                	fttm[b][i,j] = 0.25 - 0.25*exp(-4*blens[b]/3);
	        }
	        fttm[b][i,i] = 0.25 + 0.75*exp(-4*blens[b]/3);
	    }
    	}

	// copy tip data into node probability vector
	for( n in 1:S ) {
		for( i in 1:L ) {
			for( a in 1:4 ) {
				partials[n,i][a] = tipdata[n,i,a];
			}
		}
	}
    
	// calculate tree likelihood for the topology encoded in peel
	for( i in 1:L ) {
		for( n in 1:(S-1) ) {
			partials[peel[n,3],i] = (fttm[peel[n,1]]*partials[peel[n,1],i]) .* (fttm[peel[n,2]]*partials[peel[n,2],i]);
		}

		// multiply by background nt freqs (assuming uniform here)
		for(j in 1:4){
			partials[2*S,i][j] = partials[peel[S-1,3],i][j] * 0.25;
		}
		// add the site log likelihood
		target += log(sum(partials[2*S,i]));
	}
}

