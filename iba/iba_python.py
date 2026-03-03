import numpy as np

def iba(nwk, nodes, source, beta, mu, start_t, end_t, directed=False):
    n = len(nodes)
    S = np.ones(n)
    I = np.zeros(n)
    R = np.zeros(n)
    S[source] = 0.0
    I[source] = 1.0

    for t in range(start_t + 1, end_t + 1):
        # get contacts at this time
        contacts = nwk[nwk[:, 2] == t][:, 0:2]

        # compute message of each node at this time (equation 2)
        msg = 1 - beta * (1 - mu) * I

        # compute product of received neighbor messages
        get_msg = np.ones(n)
        for u, v in contacts:
            get_msg[v] *= msg[u]
            if not directed:
                get_msg[u] *= msg[v]

        # update probabilities of S, I, R
        # remark: update I first because it depends on S before S is updated
        I = (1 - mu) * I + (1 - get_msg) * S # equation 4
        S = S * get_msg  # equation 3
        R = 1 - S - I    # equation 5

    return S, I, R
