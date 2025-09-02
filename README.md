# Deep-learning-forecast-Brazil

This github has the python files used in the internship about precipitation forecasting in the northen-east coast of Brazil, and in a precise pixel using deep learning models.


# Models

Folder that contains the different deep learning models and mathematical methods used throughout the intership.

## 1. Binary classification
### Purpose
This script trains and evaluates a binary classifier to predict rain (1) vs no rain (0) at a specific latitude/longitude point, using meteorological variables from NetCDF files. The model is a simple feedforward neural network (MLP) built with PyTorch.

### How it works
* Loads .nc files containing climate data (precipitation + other variables).
* Extracts features around the chosen target coordinates (TARGET_LAT, TARGET_LON).
* Prepares input sequences of a given length (n_steps).
* Converts precipitation into a binary label using a threshold (RAIN_THRESHOLD).
* Trains an MLP with fully connected layers and evaluates it on train/validation/test sets.
* Reports metrics (accuracy, precision, recall, F1) and displays confusion matrices + loss curves.

### Usage
1. Organize your data into directories:
* data/train
* data/val
* data/test

2. Adjust the global parameters at the top of the script depending on the need.

3. Run the script, with the outputs.




