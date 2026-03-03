// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

#include "iba.h"

GLOBALS g;
NODE *n;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// this routine runs IBA for one outbreak starting from source node "source"

void iba (unsigned int source, FILE *fS, FILE *fI, FILE *fR) {
    unsigned int i, t;
    unsigned int u, v;

	// initialize probabilities
    for (i = 0; i < g.n; i++) {
        n[i].s = 1.0;
        n[i].i = 0.0;
        n[i].r = 0.0;
    }

    // set probability of source i to 1.0 and s to 0.0
    n[source].s = 0.0;
    n[source].i = 1.0;

    // loop over time steps starting from 1
    for (t = g.start_t + 1; t <= g.end_t; t++) {

        // if there are no contacts at time t
        if (g.nc[t] == 0) {
            // then probability of being susceptible does not change
            for (i = 0; i < g.n; i++) n[i].i = (1.0 - g.mu) * n[i].i;
        } else {
            // compute current message for all nodes
            for (i = 0; i < g.n; i++) n[i].msg = 1.0 - (1.0 - g.mu) * g.beta * n[i].i;

            // initialize get_msg to 1.0 for all nodes
            for (i = 0; i < g.n; i++) n[i].get_msg = 1.0;

            // loop over all contacts at time t to compute get_msg
            for (i = 0; i < g.nc[t]; i++) {
                // get nodes u and v in contact
                u = g.cl[t][i].u;
                v = g.cl[t][i].v;

                // message from u to v
                n[v].get_msg *= n[u].msg;

                if (g.dir == 0) { // if undirected
                    // also message from v to u
                    n[u].get_msg *= n[v].msg;
                }
            }

            // compute probabilities of being susceptible/infected
            for (i = 0; i < g.n; i++) {
                n[i].i = (1.0 - g.mu) * n[i].i + n[i].s * (1.0 - n[i].get_msg);
                n[i].s = n[i].s * n[i].get_msg;
            }
        }
    }

    // compute probability of recovery for all nodes
    for (i = 0; i < g.n; i++) n[i].r = 1.0 - n[i].s - n[i].i;

    // write probabilities as doubles
    double buf[g.n];
    for (i = 0; i < g.n; i++) buf[i] = fmax(log(n[i].s), -1e300);
    fwrite(buf, sizeof(double), g.n, fS);

    for (i = 0; i < g.n; i++) buf[i] = fmax(log(n[i].i), -1e300);
    fwrite(buf, sizeof(double), g.n, fI);

    for (i = 0; i < g.n; i++) buf[i] = fmax(log(n[i].r), -1e300);
    fwrite(buf, sizeof(double), g.n, fR);

}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// main function handling i/o

int main (int argc, char *argv[]) {
	unsigned int i, t;
	char *pathS, *pathI, *pathR, *log_path;

	// read network
	read_data();

	// initialize parameters
	g.beta = atof(argv[1]); // argv[1] is the infection rate
    g.mu = atof(argv[2]); // argv[2] is the recovery rate
    g.start_t = atof(argv[3]); // argv[3] is the start time of the simulation
    g.end_t = atof(argv[4]); // argv[4] is the stop time of the simulation
    g.dir = atoi(argv[5]); // argv[5] is the flag that indicates if the network is directed (1) or not (0)
	pathS = argv[6]; // argv[6] is the path to the S output file
	pathI = argv[7]; // argv[7] is the path to the I output file
    pathR = argv[8]; // argv[8] is the path to the R output file
    log_path = argv[9]; // argv[9] is the path to the log file

	// allocating nodes
	n = calloc(g.n, sizeof(NODE));

	// open output files
	FILE *fS = fopen(pathS, "wb");
    FILE *fI = fopen(pathI, "wb");
    FILE *fR = fopen(pathR, "wb");

	// run the IBA algorithm for every node as source
	FILE *logf = fopen(log_path, "w");
    for (i = 0; i < g.n; i++) {
        fprintf(logf, "Progress: %.2f%%\n", 100.0 * (i + 1) / g.n);
        fflush(logf);
        rewind(logf);

        // run IBA with source node i
        iba(i, fS, fI, fR);
    }
    fclose(logf);

    // close output files
    fclose(fS);
    fclose(fI);
    fclose(fR);

	// cleaning up
	for (t = 0; t <= g.dur; t++) {
	    free(g.cl[t]);
	}
    free(g.cl);
    free(g.nc);
	free(n);

	return 0;
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
