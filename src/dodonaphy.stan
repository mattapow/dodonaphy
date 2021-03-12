functions {
// computes the euclidean distance between a leaf and an internal node
real euclid_distance(int node_1, int node_2, int D, real[,] locs_1, real[,] locs_2) {
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
//	dp1[1] = max(dp1);
	dp1[2] = 1.0;
	iprod = dp1[1];
//	iprod = min(dp1);
	// hyperbolic 'angle'; force numerical >= 1.0
	dp1[1] = 2*(r1^2 + r2^2 - 2*r1*r2*iprod)/((1 - r1^2)*(1 - r2^2));
	dp1[2] = 0.0;
	acosharg = 1.0 + max(dp1);
	// hyperbolic distance between points i and j
	dist = 1/sqrt(curvature) * acosh(acosharg);
	return dist+0.000000000001; // add a tiny amount to avoid zero-length branches
}

void make_peel(real[] leaf_r, real[,] leaf_dir, real[] int_r, real[,] int_dir, int[,] peel, int[] location_map);

real[] compute_branch_lengths(int S, int D, int[,] peel, int[] location_map, real[] leaf_r, real[,] leaf_dir, real[] int_r, real[,] int_dir) {
	int bcount = 2*S-2;
	real blens[bcount]; // branch lengths
	for( b in 1:(S-1) ){
		real r1;
		real directional1[D];
		real r2;
		real directional2[D];
		r2 = int_r[location_map[peel[b,3]]-S];
		directional2[] = int_dir[location_map[peel[b,3]]-S,];
		if(peel[b,1] <= S){
			// leaf to internal
			r1 = leaf_r[peel[b,1]];
			directional1 = leaf_dir[peel[b,1],];
		}else{
			// internal to internal
			r1 = int_r[location_map[peel[b,1]]-S];
			directional1 = int_dir[location_map[peel[b,1]]-S,];
		}
		blens[peel[b,1]] = hyperbolic_distance(r1, r2, directional1, directional2, 1);

		// apply the inverse transform from Matsumoto et al 2020
		blens[peel[b,1]] = log(cosh(blens[peel[b,1]]))+0.000000000001; // add a tiny amount to avoid zero-length branches

		if(peel[b,2] <= S){
			// leaf to internal
			r1 = leaf_r[peel[b,2]];
			directional1 = leaf_dir[peel[b,2],];
		}else{
			// internal to internal
			r1 = int_r[location_map[peel[b,2]]-S];
			directional1 = int_dir[location_map[peel[b,2]-S],];
		}
		blens[peel[b,2]] = hyperbolic_distance(r1, r2, directional1, directional2, 1);

		// apply the inverse transform from Matsumoto et al 2020
		blens[peel[b,2]] = log(cosh(blens[peel[b,2]]))+0.000000000001;  // add a tiny amount to avoid zero-length branches
//		print("branch lengths: ");
//		print(blens[peel[b,1]]);
//		print(", ");
//		print(blens[peel[b,2]]);
//		print("\n");
	}

	return blens;
} 


real compute_LL(int S, int L, int bcount, int D, real[,,] tipdata, real[] leaf_r, real[,] leaf_dir, real[] int_r, real[,] int_dir) {
	vector[4] partials[2*S,L];   // partial probabilities for the S tips and S-1 internal nodes
	matrix[4,4] fttm[bcount]; // finite-time transition matrices for each branch
	real blens[bcount]; // branch lengths
	int peel[S-1,3];   // list of nodes for peeling. this gets initialized via the c++ function make_peel()
	int location_map[2*S-1];   // node location map. this gets initialized via  the c++ function make_peel()
	real logprob = 0;
    
	make_peel(leaf_r, leaf_dir, int_r, int_dir, peel, location_map);
	blens = compute_branch_lengths(S, D, peel, location_map, leaf_r, leaf_dir, int_r, int_dir);

	// compute the finite time transition matrices
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
		logprob += log(sum(partials[2*S,i]));
	}
	return logprob;
}

}


data {
    int D; // embedding dimension

    int <lower=0> L;               // alignment length
    int <lower=0> S;               // number of tip sequences
    real<lower=0,upper=1> tipdata[S,L,4];

    real leaf_r[S];   // radial distance of each tip sequence in the embedding
    real leaf_dir[S,D]; // directional coordinates of each tip sequence in the embedding
}

transformed data {
    int bcount; // number of branches
    bcount = 2*S-2;
}

parameters {
    real<lower=0,upper=1> int_r[S-2]; // radial distance
    real<lower=-1,upper=1> int_dir[S-2,D]; // angles
}

model {

    // compute the phylogenetic likelihood
    // first generate a tree
    // then compute a likelihood
	target += compute_LL(S, L, bcount, D, tipdata, leaf_r, leaf_dir, int_r, int_dir);
}

generated quantities {
	int peel[S-1,3];      // tree topology 
	real blens[bcount];   // branch lengths
	{
		int location_map[2*S-1];
		make_peel(leaf_r, leaf_dir, int_r, int_dir, peel, location_map);
		blens = compute_branch_lengths(S, D, peel, location_map, leaf_r, leaf_dir, int_r, int_dir);
	}
}

