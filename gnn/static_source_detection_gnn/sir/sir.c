// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// code for SIR on networks by Petter Holme (2018)

#include "sir.h"

GLOBALS g;
NODE *n;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// this routine does the bookkeeping for an infection event

void infect (float T) {
    
    // me is the node that is getting infected (the top of the heap)
	unsigned int i, you, me = g.heap[1];
    // now is the me's current time (for the source it's 0.0)
	float t, now = n[me].time;
    
    // delete the root from the heap
	del_root();
    
    // indicate that me is not susceptible anymore
	n[me].heap = I_OR_R;
	
    // get the recovery time of me
    // g.rexp contains random numbers drawn from an exponential distribution
    // with rate g.beta and thus multiplying by g.beta produces random numbers
    // drawn from an exponential distribution with rate 1
    // (which is the recovery rate by default)
	n[me].time += g.rexp[pcg_16()] * g.beta;
	
    // update extinction time if necessary
    if (n[me].time > g.t) g.t = n[me].time;
	
    // increase number of infected nodes by 1
    g.s++;

	// go through the neighbors of the infected node . .
	for (i = 0; i < n[me].deg; i++) {
		// get neighbor you
        you = n[me].nb[i];
		// check if you has already been infected
        if (n[you].heap != I_OR_R) { // if you is S, you can be infected
            // get the infection time
			t = now + g.rexp[pcg_16()] * (1.0 / n[me].weights[i]);
            //t = now + g.rexp[pcg_16()];
            // if infection time t is smaller than me's recovery time AND
            // infection time t is smaller than a potential infection time by another neighbor AND
            // infection time t is smaller or equal to the time of inference (T)
			if ((t < n[me].time) && (t < n[you].time) && (t <= T)) {
				// ... then set you's infection time to t
                n[you].time = t;
                // ... store who it was infected by
                n[you].infby = me;
				// if not listed before, then extend the heap
                if (n[you].heap == NONE) { 
					g.heap[++g.nheap] = you;
					n[you].heap = g.nheap;
				}
				up_heap(n[you].heap); // this works bcoz the only heap relationship that can be violated is the one between you and its parent
			}
		}
	}
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// this routine runs one SIR outbreak from a random seed node

void sir (unsigned int source, float T, int8_t* states) {
	unsigned int i;
	
	g.t = 0.0;
	g.s = 0;
    g.r0 = 0;
	
	// initialize
	for (i = 0; i < g.n; i++) {
		n[i].heap = NONE;
        n[i].infby = NONE;
		n[i].time = DBL_MAX; // to a large value
	}

	// get & infect the source
	// source = pcg_32_bounded(g.n);
	n[source].time = 0.0;
	n[source].heap = 1;
	g.heap[g.nheap = 1] = source;

	// run the outbreak
	while (g.nheap) infect(T);
    
    // compute R0 of current run
    for (i = 0; i < g.n; i++) if (n[i].infby == source) g.r0++; 
	
	// get node states
	for (i = 0; i < g.n; i++) {
        // if not I or R, then print state 0 (susceptible)
        if(n[i].heap != I_OR_R) {
            states[i] = 0;
        } else {
            // if I or R and recovery before T, then print state 2 (recovered)
            if (n[i].time <= T) {
                states[i] = 2;
            // else, it must be state 1 (infected)
			} else {
                states[i] = 1;
            }
        } 
    }
    
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// main function handling input

int main (int argc, char *argv[]) {
    
	unsigned int i, j, sim_per_seed, counter = 0;
    double ss1 = 0.0, ss2 = 0.0; //ss1 is used for avg. outbreak size and ss2 is used for avg. R0
	// double st1 = 0.0, st2 = 0.0, ss1 = 0.0, ss2 = 0.0; // for averages
	FILE *fp;
	
	// just a help message
	if (argc != 8) {
		fprintf(stderr, "usage: ./sir [nwk file] [beta] [T] [samp_T] [sim_per_seed] [seed] [path]\n");
		return 1;
	}

	// read stuff
	g.beta = atof(argv[2]);
    g.T = atof(argv[3]);
    g.samp_T = atoi(argv[4]);
    sim_per_seed = atoi(argv[5]);
	g.state = (uint64_t) strtoull(argv[6], NULL, 10);
	 
	// read network data file
	fp = fopen(argv[1], "r");
	if (!fp) {
		fprintf(stderr, "can't open '%s'\n", argv[1]);
		return 1;
	}
	read_data(fp);
	fclose(fp);

	// allocating the heap (N + 1) because it's indices are 1,...,N
	g.heap = malloc((g.n + 1) * sizeof(unsigned int));
    
    // 0x10000 = 65536 (hexadecimal)
	for (i = 0; i < 0x10000; i++) {
        // based on the inverse probability transform, we can sample from
        // an exponential distribution with rate parameter g.beta with 
        // -ln(U)/g.beta with U being sampled from U(0,1)
		g.rexp[i] = -log((i + 1.0) / 0x10000) / g.beta;
        // this creates random samples of T, the upper bound is 4 * g.T
        g.rT[i] = 4 * g.T * ((i + 1.0) / 0x10000);
    }
	
	// arrays in which we store states of nodes	and labels
	int8_t* states = malloc(g.n * sizeof(int8_t));
	unsigned int* labels = malloc(g.n * sim_per_seed * sizeof(unsigned int));
	
	// initialize string for filename
	char fname1[100];
	snprintf(fname1, sizeof(fname1), "%s%s", argv[7], "/states.bin");
	
	// open output files
	FILE *fS = fopen(fname1, "wb");

	// run the simulations and summing for averages
	for (i = 0; i < g.n; i++) {
        for (j = 0; j < sim_per_seed; j++) {
            // start SIR process from source i
            // depending on g.samp_T we sample the observation time or we simply set it to g.T
            if (g.samp_T == 1) sir(i, g.rT[pcg_16()], states); else sir(i, g.T, states);
			// write states of current run to file
			fwrite(states, sizeof(int8_t), g.n, fS);
			// add true source to labels array and increment counter
			labels[counter++] = i;
            // store R0 and outbreak size
            ss1 += (double) g.s;
            ss2 += (double) g.r0;
        }
	}
	
	// close output files
    fclose(fS);
	
	// export labels
	char fname2[100]; snprintf(fname2, sizeof(fname2), "%s%s", argv[7], "/labels.bin");
	FILE *fL = fopen(fname2, "wb"); fwrite(labels, sizeof(unsigned int), g.n * sim_per_seed, fL); fclose(fL);

	// make averages
	ss1 /= (g.n * sim_per_seed);
	ss2 /= (g.n * sim_per_seed);

	// print result
	printf("Avg. outbreak size: %g\n", ss1);
    printf("Avg. R0: %g\n", ss2);

	// cleaning up
	for (i = 0; i < g.n; i++) {
        free(n[i].nb);
        free(n[i].weights);
    }
	free(n); free(g.heap); free(states); free(labels);
	 
	return 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
