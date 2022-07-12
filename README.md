# SINDy Autoencoder - PyTorch
Implementation of the Ensemble SINDy Autoencoder

## To run the model:
python3 main.py

## To run  the model from a previous checkpoint:
python3 main.py --load_cp 1

## To run the experiment (automatically loads the checkpoint):
python3 experiments.py

## cmd_line.py
cmd_line.py contains arguments that can be adjusted for the model (ie, learning rates and loss lambdas)
the easiest way to do experiments is to adjust parameters in cmd_line.py and then run python3 main.py