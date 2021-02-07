functions {

// computes the distance of two points given in the Poincare ball
real poincare_distance(real[] d1, real[] d2, int D) {
  real d1sq[D];
  real d2sq[D];
  real d1md2sq[D];
  for(i in 1:D){
    d1sq[i] = pow(d1[i],2.0);
    d2sq[i] = pow(d2[i],2.0);
    d1md2sq[i] = pow((d1[i]-d2[i]),2.0);
  }
  if(sum(d1sq) >=1 || sum(d2sq) >= 1){
    print("Error");
  }
  return(acosh(1+2*((sum(d1md2sq))/(1-sum(d1sq))/(1-sum(d2sq)))));
}

void make_peel_geodesics(real[,] leaf_locs, real[,] int_locs, int[,] peel);

real[] compute_branch_lengths(int S, int D, int[,] peel, real[,] leaf_locs, real[,] int_locs) {
	int bcount = 2*S-2;
	real blens[bcount]; // branch lengths
	for( b in 1:(S-1) ){
		real directional1[D];
		real directional2[D];
		directional2[] = int_locs[peel[b,3]-S,];
		if(peel[b,1] <= S){
			// leaf to internal
			directional1 = leaf_locs[peel[b,1],];
		}else{
			// internal to internal
			directional1 = int_locs[peel[b,1]-S,];
		}
		blens[peel[b,1]] = poincare_distance(directional1, directional2, D);

		// apply the inverse transform from Matsumoto et al 2020
		blens[peel[b,1]] = log(cosh(blens[peel[b,1]]));
//    print("dist 1:");
//    print(blens[peel[b,1]]);

		if(peel[b,2] <= S){
			// leaf to internal
			directional1 = leaf_locs[peel[b,2],];
		}else{
			// internal to internal
			directional1 = int_locs[peel[b,2]-S,];
		}
		blens[peel[b,2]] = poincare_distance(directional1, directional2, D);
//    print("dist 2:");
//    print(blens[peel[b,2]]);

		// apply the inverse transform from Matsumoto et al 2020
		blens[peel[b,2]] = log(cosh(blens[peel[b,2]]));
	}

	return blens;
} 


real compute_LL(int S, int L, int bcount, int D, real[,,] tipdata, real[,] leaf_locs) {
  real int_locs[S-1,D]; // locations in the Poincare ball
	vector[4] partials[2*S,L];   // partial probabilities for the S tips and S-1 internal nodes
	matrix[4,4] fttm[bcount]; // finite-time transition matrices for each branch
	real blens[bcount]; // branch lengths
	int peel[S-1,3];   // list of nodes for peeling. this gets initialized via the c++ function make_peel()
	real logprob = 0;
    
	make_peel_geodesics(leaf_locs, int_locs, peel);
	blens = compute_branch_lengths(S, D, peel, leaf_locs, int_locs);

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
//  print(logprob);
	return logprob;
}

}


data {
  int D; // embedding dimension

  int <lower=0> L;               // alignment length
  int <lower=0> S;               // number of tip sequences
  real<lower=0,upper=1> tipdata[S,L,4];
}

transformed data {
  int bcount; // number of branches
  bcount = 2*S-2;
}

parameters {
  real<lower=-1,upper=1> leaf_locs[S,D]; // coordinates of each tip sequence in the embedding
}

model {
  // compute the phylogenetic likelihood
  // first generate a tree
  // then compute a likelihood
	target += compute_LL(S, L, bcount, D, tipdata, leaf_locs);
}

generated quantities {
	int peel[S-1,3];      // tree topology 
	real blens[bcount];   // branch lengths
	{
    real int_locs[S-1,D]; // locations in the Poincare ball
		make_peel_geodesics(leaf_locs, int_locs, peel);
		blens = compute_branch_lengths(S, D, peel, leaf_locs, int_locs);
	}
}

