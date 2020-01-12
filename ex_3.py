import numpy as np
from scipy.special import softmax
from scipy.stats import zscore

VALID_ARGS = 4
FIRST_ARG = 1
SECOND_ARG = 2
THIRD_ARG = 3
PERCENT = 0.8
NUMBER_OF_CLASSES = 10
NUMBER_OF_PIXELS = 784
ONE_ROW = 1
ONE_COLUMN = 1
EPOCHS = 60
ETA = 0.05
NUMBER_NEURONS_HIDDEN_LAYER = 120
INITIAL_MISSING = 0
MAXIMAL_MISSING_RATE = 1
Y_FICTITIOUS = 0
MINIMAL_AVG_LOSS = 10
INITIAL_LOSS = 0


# The function read_data, reads the data from the given command files,
# Load them into array and return them
def read_data():
    train_x = np.loadtxt(sys.argv[FIRST_ARG])
    train_y = np.loadtxt(sys.argv[SECOND_ARG], dtype=int)
    test_x = np.loadtxt(sys.argv[THIRD_ARG])
    return train_x, train_y, test_x


# The function shuffle, shuffles accordingly the data and the lables
def shuffle(data, lables):
    # Shuffle accordingly the data and the lables
    randomize = np.arange(len(data))
    np.random.shuffle(randomize)
    return data[randomize], lables[randomize]


# The function create_validation_set, split the training set into training and validation sets
def create_validation_set(train_x, train_y):
    train_x, train_y = shuffle(train_x, train_y)
    num_of_rows = int(PERCENT * (len(train_x)))
    # Split the training set into training and validation set
    validation_x, validation_y = train_x[num_of_rows:, :], train_y[num_of_rows:]
    train_x, train_y = train_x[:num_of_rows, :], train_y[:num_of_rows]
    return train_x, train_y, validation_x, validation_y


# The function initialize_parameters, initialize and return the weights and the biases,
# For the first iteration. It choose randomly numbers from uniform distribution over (0,1]
def initialize_parameters():
    W1 = np.random.rand(NUMBER_NEURONS_HIDDEN_LAYER,NUMBER_OF_PIXELS)
    b1 = np.random.rand(NUMBER_NEURONS_HIDDEN_LAYER, ONE_COLUMN)
    W2 = np.random.rand(NUMBER_OF_CLASSES,NUMBER_NEURONS_HIDDEN_LAYER)
    b2 = np.random.rand(NUMBER_OF_CLASSES, ONE_COLUMN)
    # Save the parameters
    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters


# ReLU function - choose the maximal value between x and zero
def relu(x):
    return np.maximum(0, x)


# ReLU prime function, if the value in x is negative,
# It turns to zero, and 1 otherwise.
def relu_prime(x):
    return 1. * (x > 0)


# The function one_hat_y, represent the label in vector that contains zeros,
# Except from the the place [y] which will contain 1.
def one_hat_y(y):
    one_hat = np.zeros(NUMBER_OF_CLASSES)
    one_hat[y] = 1
    return one_hat


# The function calculate_loss, calculate the loss, because we are representing y as one hat,
# The loss is calculate as: -log(h2[y])
def calculate_loss(y_location, h2):
    return -1*np.log(h2[y_location])


# The function feed_forward, feed our neural network with the example x and the given parameters
def feed_forward(x, y, parameters):
    W1, b1, W2, b2 = [parameters[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    # normalize the values before preforming ReLU
    z1 = zscore(z1)
    h1 = relu(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    # save the new parameters
    forward_parameters = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in parameters:
        forward_parameters[key] = parameters[key]
    return forward_parameters


# Calculate the gradients from the end to the start,
# So we could update the values of the biases and weights, accordingly.
def back_propagation(forward_parameters):
    # load the parameters that were calculate in feed_forward
    x, y, z1, h1, z2, h2 = [forward_parameters[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2')]
    y = y.reshape(NUMBER_OF_CLASSES, ONE_COLUMN)
    dz2 = (h2 - y)           #dL/dz2
    dW2 = np.dot(dz2, h1.T)  #dL/dz2 * dz2/dw2
    db2 = dz2                #dL/dz2 * dz2/db2
    dz1 = np.dot(forward_parameters['W2'].T,(h2 - y)) * relu_prime(z1)  #  dL/dz2 * dz2/dh1 * dh1/dz1
    dW1 = np.dot(dz1, x.T)   #dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1                #dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}


# Update all the parameters using SGD with the values given from back_propagation
def update_parameters(parameters, back_prop_parameters):
    W1, b1, W2, b2 = [parameters[key] for key in ('W1', 'b1', 'W2', 'b2')]
    # load the parameters that were calculate in back_propagation
    dW1, db1, dW2, db2 = [back_prop_parameters[key] for key in ('W1', 'b1', 'W2', 'b2')]
    W1 = W1 - (ETA * dW1)
    parameters['W1'] = W1
    W2 = W2 - (ETA * dW2)
    parameters['W2'] = W2
    b1 = b1 - (ETA * db1)
    parameters['b1'] = b1
    b2 = b2 - (ETA * db2)
    parameters['b2'] = b2
    return parameters


# The function miss_rate return the miss rate of the given w on the validation set
def miss_rate(validation_x, validation_y, parameters):
    miss = INITIAL_MISSING
    for x, y in zip(validation_x, validation_y):
        x = zscore(x)
        x = np.reshape(x, (784, 1))
        forward_parameters = feed_forward(x, one_hat_y(y), parameters)
        y_hat = np.argmax(forward_parameters['h2'])
        if y_hat != y:
            miss += 1
    return miss / len(validation_x)


# The function practice_neural_network - practice the neural_network,
# Throwout number of epochs. At the end it returns the parameters that gave the minimal loos.
def practice_neural_network(train_x, train_y):
    parameters = initialize_parameters()
    best_parameters = parameters
    minimal_avg_loss = MINIMAL_AVG_LOSS
    for e in range(EPOCHS):
        loss = INITIAL_LOSS
        train_x, train_y = shuffle(train_x, train_y)
        for x, y in zip(train_x, train_y):
            x = zscore(x)
            x = np.reshape(x, (784, 1))
            forward_parameters = feed_forward(x, one_hat_y(y), parameters)
            # Calculate and save the loss for every iteration
            loss += calculate_loss(y, forward_parameters['h2'])
            back_prop_parameters = back_propagation(forward_parameters)
            parameters = update_parameters(parameters, back_prop_parameters)
        # Calculate the average loss at this epoch
        avg_loss = loss/len(train_x)
        # Check if the average loss from that epoc is smaller from the minimal loss
        if avg_loss < minimal_avg_loss:
            minimal_avg_loss = avg_loss
            # Save the parameter that gave tha minimal loss
            best_parameters = parameters
    return best_parameters


# The function prediction - predict the label for every sample from the test,
# With the biases and weights that gave the minimal loss, and save them into file.
def prediction(test_x, parameters):
    f = open("test_y", "w+")
    for x in test_x:
        x = zscore(x)
        x = np.reshape(x, (784, 1))
        forward_parameters = feed_forward(x, Y_FICTITIOUS, parameters)
        y_hat = np.argmax(forward_parameters['h2'])
        f.write(f"{y_hat}\n")
    f.close()


# Red the data from the files given in the command line.
train_x, train_y, test_x = read_data()
# Split the given data into training and validation set
train_x, train_y, validation_x, validation_y = create_validation_set(train_x, train_y)
# Practice the neural network and save the parameters that minimize the loss
best_parameters = practice_neural_network(train_x, train_y)
# Make prediction on test set
prediction(test_x, best_parameters)
