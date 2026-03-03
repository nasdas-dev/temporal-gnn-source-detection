#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdint.h>

typedef struct CONTACT {
    unsigned int u;
    unsigned int v;
} CONTACT;

typedef struct GLOBALS {
	// INPUT PARAMETERS
	double beta, mu;
	// NETWORK SPECS
	unsigned int n, dur, dir; // dur is what I call t_max in python, dir is 1 if directed 0 if not
	// EDGE LIST PER TIME
	unsigned int *nc; // number of contacts at time t
	CONTACT **cl; // contact list at time t
	// CONFIGURATION
	unsigned int start_t, end_t; // start and stop algorithm at this time
} GLOBALS;

typedef struct NODE {
    double s;
    double i;
    double r;
    double msg;
    double get_msg;
} NODE;

// misc.c
extern void read_data ();
