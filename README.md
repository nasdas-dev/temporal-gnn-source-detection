# Epidemic source detection for temporal networks

Our code is a mix of fast C code adapted from Petter Holme (https://github.com/pholme/sir) and various Python routines. For the graph neural networks we rely on PyTorch and PyTorch Geometric.

The repository has the following folders. Some of them are gitignored and must be created manually.

* `data/` (gitignored): data that is created along the way.
* `exp/`: the folder with the experiments as config.yml files as well as some reading routines.
* `gephi/`: networks for gephi visualization.
* `gnn/`: the folder with the graph neural network code.
* `iba/`: the folder with the individual-based approximation code.
* `logs/` (gitignored): the folder where the logs of C Code are stored.
* `mcsim/`: the folder Monte-Carlo based inference methods, including Soft Margin.
* `nwk/` (gitignored): this subdirectory will contain the edgelists with timestamps as well as a yaml file with the graph configuration, most notably whether it is a directed or undirected graph.
* `playg/` (gitignored): a playground for testing code snippets.
* `plots/` (gitignored): the folder where the plots are stored.
* `tsir/`: the folder with the temporal SIR code in C.
* `venv/` (gitignored): the virtual environment.

One needs to set up a virtual environment. Here are the most important commands to create, use, and delete the virtual env. in which the code is run:

* `pip install virtualenv` to install the library.
* `virtualenv -p python3 venv` to create the virtual environment.
* `source venv/bin/activate` to activate the virtual environment.
* `deactivate` to stop running the virtual env.
* `pip install numpy` to install packages.
* `pip freeze --local > requirements.txt` to create a text file with all the requirements.
* `rm -rf venv/` to delete the virtual environment.
* `pip install -r requirements.txt` to recreate the virtual environment according to the specification in the text file.

