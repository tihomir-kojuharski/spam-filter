from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np

def derivative(x):
    return x * (1.0 - x)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

X = []
Y = []

# read the training data
# with open('Train.csv') as f:
#     for line in f:
#         curr = line.split(',')
#         new_curr = [1]
#         for item in curr[:len(curr) - 1]:
#             new_curr.append(float(item))
#         X.append(new_curr)
#         Y.append([float(curr[-1])])

# X = np.load('enron_features_matrix.npy');
# Y = np.load('enron_labels.npy');

# X = np.array(X)
# X = preprocessing.scale(X) # feature scaling
# Y = np.array(Y)

features_matrix = np.array(np.load('enron_features_matrix.npy'));
features_matrix = preprocessing.scale(features_matrix) # feature scaling
labels = np.array(np.load('enron_labels.npy'));
# print(features_matrix.shape)
# print(labels.shape)
X_train, X_test, Y_train, y_test = train_test_split(features_matrix, labels, test_size=0.40)
Y_train = np.array([[item] for item in Y_train])
y_test= np.array([[item] for item in y_test])

# we have 3 layers: input layer, hidden layer and output layer
# input layer has 57 nodes (1 for each feature)
# hidden layer has 4 nodes
# output layer has 1 node
dim1 = len(X_train[0])
dim2 = 4

# randomly initialize the weight vectors
np.random.seed(1)
weight0 = 2 * np.random.random((dim1, dim2)) - 1
weight1 = 2 * np.random.random((dim2, 1)) - 1

# you can change the number of iterations
for j in range(2000):
    # first evaluate the output for each training email
    layer_0 = X_train
    layer_1 = sigmoid(np.dot(layer_0,weight0))
    layer_2 = sigmoid(np.dot(layer_1,weight1))

    # calculate the error
    layer_2_error = Y_train - layer_2

    # perform back propagation
    layer_2_delta = layer_2_error * derivative(layer_2)
    layer_1_error = layer_2_delta.dot(weight1.T)
    layer_1_delta = layer_1_error * derivative(layer_1)

    # update the weight vectors
    weight1 += layer_1.T.dot(layer_2_delta)
    weight0 += layer_0.T.dot(layer_1_delta)

    if j % 100 == 0:
        temp_layer_0 = X_test
        temp_layer_1 = sigmoid(np.dot(temp_layer_0, weight0))
        temp_layer_2 = sigmoid(np.dot(temp_layer_1, weight1))
        correct = 0

        # if the output is > 0.5, then label as spam else no spam
        for i in range(len(temp_layer_2)):
            if (temp_layer_2[i][0] > 0.5):
                temp_layer_2[i][0] = 1
            else:
                temp_layer_2[i][0] = 0

            if (temp_layer_2[i][0] == y_test[i][0]):
                correct += 1

        # printing the output
        print(j, " accuracy = ", correct * 100.0 / len(temp_layer_2))

# evaluation on the testing data
layer_0 = X_test
layer_1 = sigmoid(np.dot(layer_0,weight0))
layer_2 = sigmoid(np.dot(layer_1,weight1))
correct = 0

# if the output is > 0.5, then label as spam else no spam
for i in range(len(layer_2)):
    if (layer_2[i][0] > 0.5):
        layer_2[i][0] = 1
    else:
        layer_2[i][0] = 0

    if (layer_2[i][0] == y_test[i][0]):
        correct += 1

# printing the output
print("total = ", len(layer_2))
print("correct = ", correct)
print("accuracy = ", correct * 100.0 / len(layer_2))
