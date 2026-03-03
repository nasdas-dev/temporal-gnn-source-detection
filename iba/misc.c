#include "iba.h"

extern GLOBALS g;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// this routine reads the temporal network from the standard input
// the format is strict and unflexible (and meant to be assembled by a wrapper)
// function

void read_data () {
    unsigned int t, t_read;

	if (2 != scanf("%u %u", &g.n, &g.dur)) { // the number of nodes N and duration of the data set called t_max in python
		fprintf(stderr, "reading error 1\n");
		exit(1);
	}

	g.nc = malloc((g.dur + 1) * sizeof(unsigned int)); // number of contacts at time t
    g.cl = malloc((g.dur + 1) * sizeof(CONTACT*)); // contact list at time t

	for (t = 0; t <= g.dur; t++) { // then for every time step

		if (2 != scanf("%u %u", &t_read, &g.nc[t])) { // scan the time step and the number of contacts at that time step
			fprintf(stderr, "reading error 2\n");
			exit(1);
		}
		if (t_read != t) { // sanity check that the time step is indeed t
            fprintf(stderr, "reading error 3\n");
            exit(1);
        }

        g.cl[t] = malloc(g.nc[t] * sizeof(CONTACT));
        for (unsigned int i = 0; i < g.nc[t]; i++) { // for all contacts at that time step
            if (2 != scanf("%u %u", &g.cl[t][i].u, &g.cl[t][i].v)) { // scan the two nodes in contact
                fprintf(stderr, "reading error 4\n");
                exit(1);
            }
        }
	}
}

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
