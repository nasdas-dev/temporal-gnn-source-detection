# Graph neural network for epidemic source detection

Our code is a mix of fast C code to simulate epidemic outcomes on static networks that was originally written by Petter Holme (https://github.com/pholme/sir) and various Python routines. For the graph neural networks we rely on PyTorch and PyTorch Geometric. Moreover, we provide various batch scripts.

You need to set up a virtual environment. Here are the most important commands to create, use, and delete the virtual env. in which the code is run:

* With `pip list` you can see what is already installed.
* `python3 -m venv .venv` to create the virtual environment.
* `source .venv/bin/activate` to activate the virtual environment.
* `deactivate` to stop running the virtual env.
* `which python` to see which python installation is used.
* `pip install numpy` to install packages.
* `pip freeze --local > requirements.txt` to create a text file with all the requirements.
* `rm -rf .venv/` to delete the virtual environment.
* `pip install -r requirements.txt` to recreate the virual environment according to the specification in the text file.

The file `requirements.txt` is part of the repository and setting up the virtual env. should thus be fairly easy.

To run the code, you then need to activate the virtual env. with `source .venv/bin/activate`.

There are then different options of how to run the training code:

* `./run_once.sh` if you just want to run one training run for a given network and a given specification. In the batch script you can specify everything.
* `./run_several.sh` if you want to run several training runs for a given network with different simulations and different weight initializations.
* `./run_different_T.sh` if you want to run training for different durations T, with all other parameters fixed.
* `./run_different_N.sh` if you want to run training for different numbers of simulations N (per node), with all other parameters fixed.
* `python3 hyperparameter_tuning.py` if you want to run hyperparameter tuning with the help of the optuna library (https://optuna.org/).

To run inference, you currently have two options:

* You can call the inference python routine directly providing it with the path to a trained model (basically the output directory of the training run that is stored in `data/`). This is done by `python3 -u -m sourcedet.run_inference [name of data directory]`.
* If you want to run inference for several trained models within a directory, you can use `./make_inference.sh`.
