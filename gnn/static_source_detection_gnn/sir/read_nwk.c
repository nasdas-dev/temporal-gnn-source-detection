// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// code for SIR on networks by Petter Holme (2018)


#include "sir.h"

extern NODE *n;
extern GLOBALS g;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// reads the network, assumes an edge list with vertex label 0,N-1
// if your network has nodes with degree zero, make sure that none of them is
// the node with largest index

void read_data (FILE *fp) {
	unsigned int i, me, you, weight;

	g.n = 0;

	// scan the system size
	while (3 == fscanf(fp, "%u %u %u\n", &me, &you, &weight)) {
		if (g.n < me) g.n = me;
		if (g.n < you) g.n = you;
	}

	g.n++;

	n = calloc(g.n, sizeof(NODE));

	rewind(fp);

	// scan the degrees
	while (3 == fscanf(fp, "%u %u %u\n", &me, &you, &weight)) {
		n[me].deg++;
		n[you].deg++;
	}

	// allocate adjacency and weight lists
	for (i = 0; i < g.n; i++) {
		n[i].nb = malloc(n[i].deg * sizeof(unsigned int));
        n[i].weights = malloc(n[i].deg * sizeof(unsigned int));
		n[i].deg = 0;
	}

	rewind(fp);

	// fill adjacency lists
	while (3 == fscanf(fp, "%u %u %u\n", &me, &you, &weight)) {
        // Add neighbor for me
		n[me].nb[n[me].deg] = you;
        // Add neighbor's weight
        n[me].weights[n[me].deg] = weight;
        // Add neighbor for you (this assumes that network is undirected)
		n[you].nb[n[you].deg] = me;
        // Add neighbor's weight
        n[you].weights[n[you].deg] = weight;
        // Increment degree of me and you
        n[me].deg++;
        n[you].deg++;
	}
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
