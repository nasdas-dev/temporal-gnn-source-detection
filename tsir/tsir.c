// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// code for SIR on temporal networks by Petter Holme (2018/2020)

#include "tsir.h"

GLOBALS g;
NODE *n;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// this routine first localizes the first contact later than 'now' in t
// then picks the contact that can infect (a chain of bernouilli trials)
// among the rest of the contacts. It returns the time of the infecting contact

unsigned int contagious_contact (unsigned int *t, unsigned int nt, unsigned int now) {
	unsigned int lo = 0, mid, hi = nt - 1;

	if (t[hi] <= now) return END; // no need to search further bcoz t is sorted. Note that the bisection search depends on this line.

	// the actual bisection search
	while (lo < hi) {
		mid = (lo + hi) >> 1;
		if (t[mid] > now) hi = mid;
		else lo = mid + 1;
	}

	// get a random contact
	hi += g.rnd2inx[pcg_16()];

	if (hi >= nt) return NONE; // if the contact is too late, skip it

	// return the time of the contact
	return t[hi];
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// this routine does the book keeping for an infection event

void infect () {
	unsigned int i, you, t, me = g.heap[1];
	unsigned int now = n[me].time, duration = exptime();

	del_root(); // take the newly infected off the heap

	if (duration > 0) { // if the duration is zero, no one else can be infected
		if (duration == END) { // if mu = 0, then lambda = 0 and there is no recovery, so duration = END
            n[me].time = END; // so also set time to END
        } else {
            n[me].time += duration;
        }

		// go through the neighbors of the infected node . .
		for (i = 0; i < n[me].deg; i++) {
			you = n[me].nb[i];
			if (S(you)) { // if you is S, you can be infected
				// find the infection time of you
				t = contagious_contact(n[me].t[i], n[me].nc[i], now);
				if (t == END) break; // bcoz the sorting of nbs, we can break

				// if the infection time is before when me gets recovered,
				// and (if it was already listed for infection) before the
				// previously listed infection event, then list it
				if ((t <= n[me].time) && (t < n[you].time)) {
					n[you].time = t; // set you's infection time
					n[you].infby = me; // set who infected you
					if (n[you].heap == NONE) { // if not listed before, then extend the heap
						g.heap[++g.nheap] = you;
						n[you].heap = g.nheap;
					}
					up_heap(n[you].heap); // this works bcoz the only heap relationship that can be violated is the one between you and its parent
				}
			}
		}
	}

	g.s[g.ns++] = me; // to get the outbreak size
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// this routine runs one SIR outbreak from a random starting node

void sir (unsigned int source, FILE *fS, FILE *fI, FILE *fR) {
    unsigned int i;
	g.ns = 0;
	g.r0 = 0;
	
	// get & infect the source
	n[source].time = g.start_t;
	n[source].heap = 1;
	g.heap[g.nheap = 1] = source;

	// run the outbreak
	while (g.nheap && n[g.heap[1]].time <= g.stop_t) infect();

	// compute R0 of current run
    for (i = 0; i < g.n; i++) if (n[i].infby == source && n[i].heap == END) g.r0++;

	// print states
    int8_t buf[g.n];
    for (i = 0; i < g.n; i++) buf[i] = S(i) ? 1 : 0;   // S
    fwrite(buf, sizeof(int8_t), g.n, fS);

    for (i = 0; i < g.n; i++) buf[i] = (!S(i) && n[i].time > g.stop_t) ? 1 : 0; // I
    fwrite(buf, sizeof(int8_t), g.n, fI);

    for (i = 0; i < g.n; i++) buf[i] = (!S(i) && n[i].time <= g.stop_t) ? 1 : 0; // R
    fwrite(buf, sizeof(int8_t), g.n, fR);

	// clean
	for (i = 0; i < g.ns; i++) n[g.s[i]].heap = n[g.s[i]].time = NONE;
	for (i = 1; i <= g.nheap; i++) n[g.heap[i]].heap = n[g.heap[i]].time = NONE;
	// might be easiest to just reset the heap and time of all nodes:
	// for (i = 0; i < g.n; i++) n[i].heap = n[i].time = NONE;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// main function handling i/o

int main (int argc, char *argv[]) {
	unsigned int i, j, k, n_runs, N;
	double mu, d, x, R0 = 0.0, s1 = 0.0, s2 = 0.0, avg, dss, sd; // s1 is the sum of outbreak sizes, s2 is the sum of squares
	char *pathS, *pathI, *pathR, *log_path;

	// read network
	read_data();

	// initialize infection parameter
	d = atof(argv[1]);
	if (d < 1.0) {
		d = 1.0 / log(1.0 - d); // argv[1] is beta, the per-contact infection probability
		for (i = 0; i < 0x10000; i++) {
			x = d * log((i + 1) / 65536.0);
			g.rnd2inx[i] = (x > USHRT_MAX) ? USHRT_MAX : x;
		}
	} else for (i = 0; i < 0x10000; i++) g.rnd2inx[i] = 0;

    // initialize recovery parameter
    mu = atof(argv[2]); // argv[2] is the recovery probability per time stp
    // Note that if Y ~ Exp(lambda), then rounded_up(Y) ~ Geom(mu) where mu = 1 - exp(-lambda).
    g.lambda = -log(1.0 - mu); // compute recovery rate from the recovery probability mu

    // initialize other parameters
    g.start_t = atof(argv[3]); // argv[3] is the start time of the simulation
    g.stop_t = atof(argv[4]); // argv[4] is the stop time of the simulation
    n_runs = atoi(argv[5]); // argv[5] is the number of runs per source node
	g.state = strtoull(argv[6], NULL, 10); // argv[6] is my seed
	pathS = argv[7]; // argv[7] is the path to the S output file
	pathI = argv[8]; // argv[8] is the path to the I output file
    pathR = argv[9]; // argv[9] is the path to the R output file
	log_path = argv[10]; // argv[10] is the path to the log file

	// warm up the random number generator
	for (k = 0; k < 10; k++)
        pcg_32(); // the first numbers are often trash

	// allocating the heap (N + 1) because it's indices are 1,...,N
	g.heap = malloc((g.n + 1) * sizeof(unsigned int));
	g.s = calloc(g.n, sizeof(unsigned int));

	// initialize
	for (i = 0; i < g.n; i++) n[i].heap = n[i].time = NONE;

	// open output files
	FILE *fS = fopen(pathS, "wb");
    FILE *fI = fopen(pathI, "wb");
    FILE *fR = fopen(pathR, "wb");

	// run the simulations and summing for averages
	FILE *logf = fopen(log_path, "w");
	for (i = 0; i < g.n; i++) { // for every node as source node
        fprintf(logf, "Progress: %.2f%%\n", 100.0 * (i + 1) / g.n);
        fflush(logf);
        rewind(logf);
        for (j = 0; j < n_runs; j++) { // n_runs per source node
            sir(i, fS, fI, fR); // run the SIR simulation and pass the source node

            // saving stats for averages
            R0 += (double)g.r0;
            d = g.ns;
            s1 += d;
            s2 += SQ(d);
        }
	}
	fclose(logf);

	// close output files
    fclose(fS);
    fclose(fI);
    fclose(fR);

	// print outbreak size average and standard deviation
	N = g.n * n_runs; // total number of runs
	avg = s1 / N; // average outbreak size
	dss = s2 - 2 * avg * s1 + N * SQ(avg); // "Abweichungsquadratsumme" sum_i(x_i - avg)^2 where x_i = outbreak size of run i
	sd = sqrt(dss / (N - 1)); // standard deviation of outbreak size = sqrt(Abweichungsquadratsumme / (N - 1))
	printf("%g %g %g %g", R0 / N, avg, sd, sd / sqrt(N)); // the last is the standard error of the mean estimate
	
	// cleaning up
	for (i = 0; i < g.n; i++) {
		for (j = 0; j < n[i].deg; j++) free(n[i].t[j]);
		free(n[i].nb);
		free(n[i].nc);
		free(n[i].t);
	}
	free(n); free(g.heap); free(g.s);
	 
	return 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
