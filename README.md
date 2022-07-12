# SINDy Autoencoder - PyTorch
PyTorch implementation of the SINDy Autoencoder from the paper "Data-driven discovery of coordinates and governing equations"
by Champion et al.

The original implementation is in TensorFlow and is located at: https://github.com/kpchamp/SindyAutoencoders <br>
The TensorFlow implementation was used as a reference and, for some components, code was directly copied over.
In the latter case, each file with copied code has a reference to which file in the original repository it was taken from.

## Datasets
Currently, the model supports only using a Lorenz system dataset.
To create it, run "python3 create_lorenz.py"

## To run the model:
python3 main.py

## To run  the model from a previous checkpoint:
python3 main.py --load_cp 1

## To run the experiment (automatically loads the checkpoint):
python3 experiments.py

## cmd_line.py
cmd_line.py contains arguments that can be adjusted for the model (ie, learning rates and loss lambdas)<br>
The easiest way to do experiments is to adjust parameters in cmd_line.py and then run python3 main.py.