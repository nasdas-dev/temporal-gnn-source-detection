from read_run import tsir, make_c_readable_from_nparray

# 1. Prepare your network data (H_array is a Nx3 numpy array of edges [u, v, t])
# end_t should be the max simulation time
H_cedges = make_c_readable_from_nparray(H_array, end_t, n_nodes)

# 2. Define your configuration
# The wrapper expects keys like config["sir"]["beta"], config["sir"]["mu"], etc.
tsir_config = {
    "sir": {
        "beta": 0.05,
        "mu": 0.01,
        "start_t": 0,
        "end_t": 50,
        "n_runs": 1000
    },
    "nwk": {
        "directed": False
    }
}

# 3. Run the wrapper
# This will return the log-probabilities for S, I, and R states
tsir_S, tsir_I, tsir_R = tsir(tsir_config, H_cedges, n_nodes)