from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_prime(x):
    return x * (1.0 - x)

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

class NeuralNetwork:
    """docstring for NeuralNetwork"""
    def __init__(self, activation='sigmoid'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        self.weight0 = []
        self.weight1 = []

    def fit(self, X_train, Y_train, learning_rate=0.2, epochs=3000):
        Y_train = np.array([[item] for item in Y_train])

        # we have 3 layers: input layer, hidden layer and output layer
        # input layer has 57 nodes (1 for each feature)
        # hidden layer has 4 nodes
        # output layer has 1 node
        dim1 = len(X_train[0])
        dim2 = 4

        # randomly initialize the weight vectors
        np.random.seed(1)
        self.weight0 = 2 * np.random.random((dim1, dim2)) - 1
        self.weight1 = 2 * np.random.random((dim2, 1)) - 1

        # you can change the number of iterations
        for j in range(epochs):
            # print('epoch:', j)
            # first evaluate the output for each training email
            layer_0 = X_train
            layer_1 = self.activation(np.dot(layer_0, self.weight0))
            layer_2 = self.activation(np.dot(layer_1, self.weight1))

            # calculate the error
            layer_2_error = Y_train - layer_2

            # perform back propagation
            layer_2_delta = layer_2_error * self.activation_prime(layer_2) * learning_rate
            layer_1_error = layer_2_delta.dot(self.weight1.T)
            layer_1_delta = layer_1_error * self.activation_prime(layer_1) * learning_rate

            # update the weight vectors
            self.weight1 += layer_1.T.dot(layer_2_delta)
            self.weight0 += layer_0.T.dot(layer_1_delta)

            if j % 100 == 0 and len(self.X_test) > 0 and len(self.Y_test) > 0:
                temp_layer_0 = self.X_test
                temp_layer_1 = self.activation(np.dot(temp_layer_0, self.weight0))
                temp_layer_2 = self.activation(np.dot(temp_layer_1, self.weight1))
                correct = 0

                # if the output is > 0.5, then label as spam else no spam
                for i in range(len(temp_layer_2)):
                    if (temp_layer_2[i][0] > 0.5):
                        temp_layer_2[i][0] = 1
                    else:
                        temp_layer_2[i][0] = 0

                    if (temp_layer_2[i][0] == self.Y_test[i]):
                        correct += 1

                # printing the output
                print(j, " accuracy = ", round(correct * 100.0 / len(temp_layer_2), 2))

    def predict(self, X_test):
        # evaluation on the testing data
        layer_0 = X_test
        layer_1 = self.activation(np.dot(layer_0, self.weight0))
        layer_2 = self.activation(np.dot(layer_1, self.weight1))

        # if the output is > 0.5, then label as spam else no spam
        for i in range(len(layer_2)):
            if (layer_2[i][0] > 0.5):
                layer_2[i][0] = 1
            else:
                layer_2[i][0] = 0

        return layer_2

if __name__ == '__main__':

    nn = NeuralNetwork(activation='sigmoid')
    X = []
    Y = []

    features_matrix = np.array(np.load('enron_features_matrix.npy'))
    features_matrix = preprocessing.scale(features_matrix) # feature scaling
    labels = np.array(np.load('enron_labels.npy'))
    # print(features_matrix.shape)
    # print(labels.shape)
    X_train, X_test, Y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)

    y_test = np.array([[item] for item in y_test])
    nn.fit(X_train, Y_train, epochs=1000)

    result = nn.predict(X_test)

    print("==================\n{0}:\n{1}".format(str(nn.__class__.__name__), confusion_matrix(y_test, result)))


