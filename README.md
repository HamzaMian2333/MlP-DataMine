#RF Interference Classification with an MLP
This is our research project for Purdue's Data Mine Corporate Partners Program with The Aerospace Corporation. We are using a multilayer perceptron (MLP), a type of neural network, to classify interference in GPS/GNSS signal data.

The goal is to see how well a simple neural network can tell different types of interference apart.
#Data
The project uses the Fraunhofer Highway2 dataset and Aerospace's Radio Mana library to load the data.

Each sample is a 512 x 243 power spectral density (PSD) spectrogram. There are nine classes: four background classes, three chirp jammer distance classes, and two cigarette-lighter interference classes. The dataset is not included in this repo.
What the code does
- Resizes each spectrogram to 32 x 64 and flattens it into 2,048 inputs.
- Trains an MLP with three hidden layers: 512, 256, and 128 neurons.
- Uses ReLU, cross-entropy loss, and the Adam optimizer.
- Tracks validation accuracy, macro F1, and F1 for each class.
- Saves graphs to help compare performance during training.
There is also a separate experiment in preprocess.py that uses the mean, standard deviation, maximum, and minimum at each frequency. It uses training statistics to standardize the features. The current training scripts do not use this version yet.
Results so far
These numbers come from the saved mlp_history.json file:
Result	Value
Best validation macro F1	0.698 at epoch 22
Validation accuracy at that epoch	76.8%
Highest validation accuracy	78.1% at epoch 29


These are validation results from an earlier run. The exact model and preprocessing settings for that run still need to be documented so it can be repeated.

The model struggles more with some background classes. Training loss keeps falling while validation loss eventually rises, which suggests overfitting.

Files
File	What it does
my_mlp.py	Defines the model used by the training scripts
train_mlp.py	Runs training for 20 epochs
train_mlp_with_graphs.py	Runs 40 epochs, saves metrics, graphs, and the best validation-F1 checkpoint
preprocess.py	Creates statistical features for a separate experiment
mlp.py	Defines another MLP with batch normalization and dropout
plot_training.py	Makes presentation plots; needs updates to work with the current scripts
mlp_history.json	Stores results from an earlier training run
plots/	Contains saved figures


Still working on
- Connecting the different experiments into one clear workflow.
- Updating the plotting script to use the current model files and results.
- Improving performance on the smaller classes.
- Testing only the final selected model instead of checking test accuracy every epoch.
- Making plot labels match the actual results; one older annotation incorrectly says background d is never predicted correctly.
