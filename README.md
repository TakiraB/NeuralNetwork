# Neural Network from Scratch

Welcome to the Neural Network from Scratch project! This repository contains a Python-based neural network built entirely using NumPy. The aim is to provide an educational tool for understanding the fundamentals of neural networks, including their architecture, forward propagation, backpropagation, and how they learn.

## Project Status: In Progress 🚧

**Important**: This project is in an ongoing development phase. The neural network's core functionality, such as forward propagation, backpropagation, and loss calculations, is in place. However, we've identified an issue where the model reports abnormally high accuracy, indicating a potential flaw in the backpropagation logic or the accuracy calculation method. Efforts are underway to diagnose and resolve this issue.

## Getting Started

To use this neural network, you'll need a Python environment with NumPy installed. The network is configured and executed via command line arguments, offering flexibility in experimenting with various neural network parameters.

### Prerequisites

- Python 3.6 or higher
- NumPy

If you don't have NumPy installed, you can install it using pip:

pip install numpy

### Usage

The neural network script accepts several command line arguments for configuration:

python neural_network.py -train_feat <train_features_file> -train_target <train_targets_file> -dev_feat <dev_features_file> -dev_target <dev_targets_file> -epochs <number_of_epochs> -learnrate <learning_rate> -nunits <number_of_hidden_units> -type <problem_type> -hidden_act <hidden_activation_function> -init_range <initialization_range> -num_classes <number_of_classes>

Arguments:

- `-train_feat`: Path to the training features file.
- `-train_target`: Path to the training targets file.
- `-dev_feat`: Path to the development (validation) features file.
- `-dev_target`: Path to the development (validation) targets file.
- `-epochs`: Number of epochs for training.
- `-learnrate`: Learning rate for gradient descent.
- `-nunits`: Number of units in the hidden layer.
- `-type`: Problem type, 'C' for classification or 'R' for regression.
- `-hidden_act`: Activation function for the hidden layer ('sig' for sigmoid, 'tanh' for hyperbolic tangent, 'relu' for rectified linear unit).
- `-init_range`: Range for weight initialization.
- `-num_classes`: Number of classes (for classification problems only).

