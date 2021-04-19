import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from DeepLearning.get_data import load_mnist
from DeepLearning.functions import *


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    # trained weight
    with open("DeepLearning/sample_weight.pkl", 'rb') as file:
        network = pickle.load(file)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)
    return y

# get mnist dataset
x, t = get_data()
# initialize network
network = init_network()

batch_size = 100  # batch size
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    # x[0,100], x[100:200], x[100,200]...
    x_batch = x[i:i+batch_size]
    # categorize with predict()
    # it returns the percentage of each label in numpy array <ex>[0.1, 0.3...0.04]
    y_batch = predict(network, x_batch)
    # find the largest number's index in y_batch
    p = np.argmax(y_batch, axis=1)
    # get accuracy
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))