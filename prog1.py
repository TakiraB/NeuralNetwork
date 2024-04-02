import numpy as np
import argparse

#Parsing arguments from the command line. Flags that have been ommitted are verbose mode, number of hidden layers, and minibatch size.
parser = argparse.ArgumentParser(description="Building a Neural Network from Scratch")
parser.add_argument('-train_feat', dest='train_feat_fn', required=True)
parser.add_argument('-train_target', dest='train_target_fn')
parser.add_argument('-dev_feat', dest='dev_feat_fn')
parser.add_argument('-dev_target', dest='dev_target_fn')
parser.add_argument('-epochs', dest='epochs', type=int)
parser.add_argument('-learnrate', dest='learnrate', type=float)
parser.add_argument('-nunits', dest='num_hidden_units', type=int)
parser.add_argument('-type', dest='problem_mode', choices=['C', 'R'], type=str)
parser.add_argument('-hidden_act', dest='hidden_unit_activation', choices=['sig', 'tanh', 'relu'], type=str)
parser.add_argument('-init_range', dest='init_range', type=float)
parser.add_argument('-num_classes', dest='C', type=int)
args = parser.parse_args()



def one_hot_encode(labels, num_classes):
    identity_matrix = np.eye(num_classes)
    one_hot_encoded = identity_matrix[labels]
    return one_hot_encoded


#Initialize the weights of our model. Since we are dealing with neurons and classes, these directly affect the sizes of
#our weights and biases. The number of classes affects the output weight and output bias dimensions. If we have 2 classes, we will be
#predicting probabilities on 2 classes, 3 classes and 3 probabilities, etc. For Regression, the number of classes affects the output
#dimensions of both our output weights and output biases.
def init_weights(input_size, output_size, range, hidden_units, num_classes):
    #w1
    W1_before = np.random.uniform(low= -range, high=range, size=(input_size, hidden_units))
    #b1
    B1_before = np.random.uniform(low= -range, high=range, size=(1, hidden_units))
    #w2
    W2_before = np.random.uniform(low= -range, high=range, size=(hidden_units, num_classes))
    #b2
    B2_before = np.random.uniform(low= -range, high=range, size=(1, num_classes))

    return W1_before, B1_before, W2_before, B2_before

#Forward propagation function - Compute the linear transformation of the hidden layer by dot producting the input with our hidden weights,
#and then adding our bias. We take that output (z1) and feed it into our hidden activation function to get (a1). We then taken our (a1)
#and perform the linear transformation of the output layer, dotting our (z1) wih our output weights, adding our output bias.
#After this, the output activation function is determined by the type of task we are performing, Classification or Regression.
#If Regression, we are only using the identity function. If Classification and we have two classes, we will perform sigmoid with binary
#classification. Else (we have more classes), then we will perform Softmax with multivariate classification. 
def Forward(input, W1_before, B1_before, W2_before, B2_before, hidden_activation, problem_mode, num_classes):
#------------------Hidden Layer
    #z1
    linear_transform_hidden = np.dot(input, W1_before) + B1_before
    #a1
    hidden_output = hidden_layer_activation(linear_transform_hidden, hidden_activation, Derivative=False)

#------------------Output Layer
    #z2
    linear_transform_output = np.dot(hidden_output, W2_before) + B2_before

    #Calculation of a2 based on user-chosen hidden activation function
    if problem_mode == 'R':
        forward_output = identity(linear_transform_output)
    elif problem_mode == 'C':
        if num_classes == 2:
            forward_output = sigmoid(linear_transform_output)
        else:
            forward_output = softmax(linear_transform_output)

    return linear_transform_hidden, hidden_output, linear_transform_output, forward_output

#Compute the Loss for both of our training and dev sets. For Classification, we use the accuracy performance metric, which measures
#the number of points correctly predicted divided by the total number of points (so N rows). If it's Regression, we calculate the Loss
#using MSE and we use this as our performance metric.
# Ensure Compute_Loss function works with one-hot encoded labels
def Compute_Loss(input, output, forward_output, problem_mode):
    x_shape = input.shape[0]
    if problem_mode == 'C':
        winning_class = np.argmax(forward_output, axis=1)
        true_classes = np.argmax(output, axis=1)  # Modify this line to work with one-hot labels
        correct_predictions = np.sum(winning_class == true_classes)
        accuracy = correct_predictions / x_shape
        return accuracy
    else:
        loss = (1 / x_shape) * np.sum((forward_output - output) ** 2)
        return loss

#Backward Propagation function (I decided to explain things step-by-step for my sake and extra clarity)
def Backward(input, W1_before, W2_before, linear_transform_hidden, forward_output, true_output, activation_function, hidden_output):

    size = input.shape[0]
#------------------Output Layer
    #Repeat the following for each layer: 3 steps!

    #Initialize delta of error signal (model output - true output) but we need to use broadcasting to make the dimensions match up.
    dl_dz2 = forward_output - true_output[:, None]

    #Gradient of Loss w/ respect to output weight matrix, divided by size N (I was getting crazy large gradients)
    grad_weights2 = np.dot(hidden_output.T, dl_dz2)/size

    #Gradient of Loss w/ respect to output bias vector, summed over the entire dimension N, divided by size N
    grad_bias2 = np.sum(dl_dz2, axis=0, keepdims=True)/size

#------------------Input Layer

    #Produce the error signal at previous layer = f'l-1(evaluating at pre-activation) element wise multiplying by weights of layer L 
    #and the error signal of layer L
    hidden_deriv = hidden_layer_activation(linear_transform_hidden, activation_function, Derivative=True)

    #Repeat the 3-step process over again for the hidden layer, and find delta of the error signal to proceed with the gradients of the hidden layer
    dl_dz1 = np.dot(dl_dz2, W2_before.T) * hidden_deriv

    #Since we are moving backwards, we will dot product our error signal with our input to get the gradient of the weight matrix
    grad_weights1 = np.dot(input.T, dl_dz1)/size

    #Finding the gradients of the bias vector in the hidden layer
    grad_bias1 = np.sum(dl_dz1, axis=0, keepdims=True)/size

    return grad_weights2, grad_bias2, grad_weights1, grad_bias1

#Update weights using gradient descent. This takes the original weight matrices and bias vectors and subtracts the gradients we found in our back propagation
#multiplied by the learnrate given in the command line args. For most of my testing, I used values of 0.1, 0.01 and 0.001 for the learning rate.
# def Update_weights (grad_weights2, grad_bias2, grad_weights1, grad_bias1, W1_before, B1_before, W2_before, B2_before, learnrate):
def Update_weights (grad_weights2, grad_bias2, grad_weights1, grad_bias1, W1_before, B1_before, W2_before, B2_before, learnrate):

    W1 = W1_before - learnrate * grad_weights1
    B1 = B1_before - learnrate * grad_bias1
    W2 = W2_before - learnrate * grad_weights2
    B2 = B2_before - learnrate * grad_bias2

    return W1, B1, W2, B2

#-------------Function Declarations + Conversion from Argparse
#Each of these functions has a Derivative=False tied to each. They default to False when called without the flag. If set to True, the derivative of that function is used
#instead of the regular functionality

#Sigmoid hidden activation function + output activation function for binary classification
def sigmoid(x, Derivative=False):
    sig = 1 / (1 + np.exp(-x))  # This calculates the sigmoid function
    if Derivative:
        return sig * (1 - sig)  # This uses the already computed sigmoid value to calculate the derivative
    else:
        return sig


#Tanh hidden activation function
def tanh(x, Derivative=False):
    if Derivative:
        return 1-np.tanh(x)**2
    else:
        return np.tanh(x)

#ReLu hidden activation function
# ReLu hidden activation function
def relu(x, Derivative=False):
    if Derivative:
        return (x > 0).astype(float)  # This applies the condition element-wise and returns 1.0 for x > 0 and 0.0 for x <= 0
    else:
        return np.maximum(x, 0)  # This applies ReLU element-wise


#Softmax output activation function - Diego informed me that I don't actually need the derivative of Softmax for the assignment, so I have set the
#Derivative=True to return nothing. It is also never called with the Derivative=True flag anywhere in the program.
def softmax(x, Derivative=False):
    if Derivative:
        return None
    else:
        return np.exp(x)/np.sum(np.exp(x), axis=1, keepdims=True)

#Identity output activation function (only used for Regression)
def identity(x, Derivative=False):
    if Derivative:
        return 1
    else:
        return x

#Function does a couple things at the same time. It takes in an input, activate_function and boolean flag parameter. Activate_function is initially a string such as
#'sig', 'tanh', or 'relu' from the command line and matches the string with the appropriate activation function. It is also able to process the input params at the same
#time and create Z1 in forward propagation simultaneously. 
def hidden_layer_activation(input, activate_function, Derivative=False):
    if activate_function == 'sig':
        return sigmoid(input, Derivative=False)
    if activate_function == 'tanh':
        return tanh(input, Derivative=False)
    if activate_function == 'relu':
        return relu(input, Derivative=False)
    
#-------------Reading in Files based on Assignment Requirements

#Reading the files based on Classification or Regression mode
if args.problem_mode == 'C':
    train_features = np.loadtxt(args.train_feat_fn, dtype=int)
    dev_features = np.loadtxt(args.dev_feat_fn, dtype=int)
    train_labels = np.loadtxt(args.train_target_fn, dtype=int)
    dev_labels = np.loadtxt(args.dev_target_fn, dtype=int)

    # Perform one-hot encoding for classification labels
    num_classes = args.C  # Assuming this is correctly set to the number of classes
    train_labels = one_hot_encode(train_labels, num_classes)
    dev_labels = one_hot_encode(dev_labels, num_classes)

# Regression mode (I originally had delimiters in here via the instructions but the MATLAB test regression code did not like it)
else:
    train_features = np.loadtxt(args.train_feat_fn, dtype=float)
    dev_features = np.loadtxt(args.dev_feat_fn, dtype=float)
    train_labels = np.loadtxt(args.train_target_fn, dtype=float)
    dev_labels = np.loadtxt(args.dev_target_fn, dtype=float)

#-------------Initializations

#Initializations for our weights and biases
input_dims = train_features.shape[1]
output_dims = train_labels.shape[0]
hidden_units = args.num_hidden_units
classes = args.C

#Call our init_weights function to initialize our weights and biases for the main training loop
W1_before, B1_before, W2_before, B2_before = init_weights(input_dims, output_dims, args.init_range, hidden_units, classes)

#Grab our string from the command line args so we can process our hidden activation function appropriately
hidden_function = args.hidden_unit_activation

#Grab our epochs from the command line args to determine the amount of loops we will make in training
epoch = args.epochs

#Grabbing our learnrate from the command line args to apply as hyperparameter to our update function
learnrate = args.learnrate

#Grabbing our problem mode from the command line args to modify our task
problem_mode = args.problem_mode

#-------------Training Loop

#Loop through based on the amount of epochs given - no minibatching, so no need for an inner loop
#Loop our training set through Foward Prop, grab the Loss on training for printing later, grab the Loss on dev for printing later with our predictions from the
#training set, grab the gradients of our hidden weights and biases and output weights and biases which we then pass into our Update_weights function to adjust the weights
#and biases. We then print for each epoch, the training and dev loss using Python f-string formatting (the best).
for epoch in range(epoch):

    linear_transform_hidden, hidden_output, linear_transform_output, forward_output = Forward(train_features, W1_before, B1_before, W2_before, B2_before, hidden_function, problem_mode, classes)

    Train_loss = Compute_Loss(train_features, train_labels, forward_output, problem_mode)

    Dev_loss = Compute_Loss(dev_features, dev_labels, forward_output, problem_mode)

    grad_weights2, grad_bias2, grad_weights1, grad_bias1 = Backward(train_features, W1_before, W2_before, linear_transform_hidden, forward_output, train_labels, hidden_function, hidden_output)

    W1_before, B1_before, W2_before, B2_before = Update_weights(grad_weights2, grad_bias2, grad_weights1, grad_bias1, W1_before, B1_before, W2_before, B2_before, learnrate)
    
    print(f'Epoch {epoch:03d}: train={Train_loss:.3f} dev={Dev_loss:.3f}')
