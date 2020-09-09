#################################
# Your name: Shay fux
#################################

# Please import and use stuff only from the packages numpy, sklearn, matplotlib

import numpy as np
import numpy.random
from sklearn.datasets import fetch_openml
import sklearn.preprocessing
import matplotlib.pyplot as plt

"""
Assignment 3 question 2 skeleton.

Please use the provided function signature for the SGD implementation.
Feel free to add functions and other code, and submit this file with the name sgd.py
"""


def helper_hinge():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    neg, pos = "0", "8"
    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:60000] == neg) | (labels[:60000] == pos))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[60000:] == neg) | (labels[60000:] == pos))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = (labels[train_idx[:6000]] == pos) * 2 - 1

    validation_data_unscaled = data[train_idx[6000:], :].astype(float)
    validation_labels = (labels[train_idx[6000:]] == pos) * 2 - 1

    test_data_unscaled = data[60000 + test_idx, :].astype(float)
    test_labels = (labels[60000 + test_idx] == pos) * 2 - 1

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


def helper_ce():
    mnist = fetch_openml('mnist_784')
    data = mnist['data']
    labels = mnist['target']

    train_idx = numpy.random.RandomState(0).permutation(np.where((labels[:8000] != 'a'))[0])
    test_idx = numpy.random.RandomState(0).permutation(np.where((labels[8000:10000] != 'a'))[0])

    train_data_unscaled = data[train_idx[:6000], :].astype(float)
    train_labels = labels[train_idx[:6000]]

    validation_data_unscaled = data[train_idx[6000:8000], :].astype(float)
    validation_labels = labels[train_idx[6000:8000]]

    test_data_unscaled = data[8000 + test_idx, :].astype(float)
    test_labels = labels[8000 + test_idx]

    # Preprocessing
    train_data = sklearn.preprocessing.scale(train_data_unscaled, axis=0, with_std=False)
    validation_data = sklearn.preprocessing.scale(validation_data_unscaled, axis=0, with_std=False)
    test_data = sklearn.preprocessing.scale(test_data_unscaled, axis=0, with_std=False)
    return train_data, train_labels, validation_data, validation_labels, test_data, test_labels


# Q1
def sign(w, x):  # w and x are nd.arrays
    dot_product = np.dot(w, x)
    if dot_product >= 0:
        return 1
    else:
        return -1


def accuracy_precentage(w, data, labels):
    counter = 0
    n = len(data)
    for i in range(n):
        x = np.array(data[i])
        predictor = sign(w, x)
        if predictor == labels[i]:
            counter = counter + 1
    return (counter / n)


def SGD_hinge(data, labels, C, eta_0, T):
    """
    Implements Hinge loss using SGD.
    """
    n = len(data) - 1
    w = [0 for i in range(len(data[0]))]
    w = numpy.array(w, dtype=numpy.float64)
    for t in range(1, T + 1):  # run T iterations
        eta = (eta_0 / t)
        i = np.random.random_integers(0, n)
        if labels[i] * np.dot(data[i], w) > 1:
            w = (1 - eta) * w
        else:
            w = (1 - eta) * w + C * eta * labels[i] * data[i]
    return w


# (a)
def best_ate():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    eta_lst = [pow(10, i) for i in range(-5, 6)]
    avg_accuracy_lst = []
    for eta_0 in eta_lst:
        accuracy = 0
        for t in range(10):
            w = SGD_hinge(train_data, train_labels, 1, eta_0, 1000)
            accuracy += accuracy_precentage(w, validation_data, validation_labels)
        avg_accuracy_lst.append(accuracy / 10)
    plt.plot(eta_lst, avg_accuracy_lst)
    plt.xscale('log')
    plt.xlabel('eta0 values')
    plt.ylabel('average accuracy on validation_set')
    # plt.show()


# best_ate()

# (b)
def best_C():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    C_lst = [pow(10, i) for i in range(-5, 6)]
    avg_accuracy_lst = []
    for C in C_lst:
        accuracy = 0
        for t in range(10):
            w = SGD_hinge(train_data, train_labels, C, 1, 1000)
            accuracy += accuracy_precentage(w, validation_data, validation_labels)
        avg_accuracy_lst.append(accuracy / 10)
    plt.plot(C_lst, avg_accuracy_lst)
    plt.xscale('log')
    plt.xlabel('C values')
    plt.ylabel('average accuracy on validation_set')
    # plt.show()


# best_C()

# (c)
def best_C_and_eta0():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    w = SGD_hinge(train_data, train_labels, pow(10, -4), 1, 20000)
    plt.imshow(np.reshape(w, (28, 28)), interpolation='nearest')
    # plt.show()


# best_C_and_eta0()

# (d)
def best_classifier_accuracy():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_hinge()
    w = SGD_hinge(train_data, train_labels, pow(10, -4), 1, 20000)
    test_accuracy = accuracy_precentage(w, test_data, test_labels)
    # print(test_accuracy)


# best_classifier_accuracy()

# Q2
def dotlst_withoutmax_calculator(w_matrix, x):
    lst = np.array([np.dot(x, w_matrix[i]) for i in range(len(w_matrix))])
    lst = lst - max(lst)
    return lst


def softmax_calculator(z):
    ex = np.exp(z)
    es = np.sum(ex)
    return ex / es


def gradient_calculator(x, w_matrix, y):
    z = dotlst_withoutmax_calculator(w_matrix, x)
    soft_max_vector = softmax_calculator(z)
    soft_max_vector[int(y)] = soft_max_vector[int(y)] - 1  # - Indicator
    gradient_matrix = np.array([soft_max_vector[i] * x for i in range(len(soft_max_vector))])
    return gradient_matrix


def max_classifier(w_matrix, x):
    dot_product_lst = [numpy.dot(w_matrix[i], x) for i in range(len(w_matrix))]
    return dot_product_lst.index(max(dot_product_lst))


def accuracy_precentage_v2(w_matrix, data, labels):
    counter = 0
    n = len(data)
    for i in range(n):
        x = np.array(data[i])
        predictor = max_classifier(w_matrix, x)
        if str(predictor) == labels[i]:
            counter = counter + 1
    return (counter / n)


def SGD_ce(data, labels, eta_0, T):
    """
    Implements multi-class cross entropy loss using SGD.
    """
    d = len(data[0])
    n = len(data) - 1
    w_matrix = numpy.array([[0 for i in range(d)] for j in range(10)])
    for t in range(1, T + 1):
        i = np.random.random_integers(0, n)
        w_matrix = w_matrix - eta_0 * gradient_calculator(data[i], w_matrix, labels[i])
    return w_matrix


# (a)
def best_eta_entropy():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    eta_lst = [pow(10, i) for i in range(-5, 6)]
    avg_accuracy_lst = []
    for eta_0 in eta_lst:
        accuracy = 0
        for t in range(10):
            w_matrix = SGD_ce(train_data, train_labels, eta_0, 1000)
            accuracy += accuracy_precentage_v2(w_matrix, validation_data, validation_labels)
        avg_accuracy_lst.append(accuracy / 10)
    plt.plot(eta_lst, avg_accuracy_lst)
    plt.xscale('log')
    plt.xlabel('eta0 values')
    plt.ylabel('average accuracy on validation_set')
    # plt.show()


# best_eta_entropy()

# (b)


def best_ce_eta_classifier():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    w_matrix = SGD_ce(train_data, train_labels, pow(10, -2), 20000)

    fig, ax = plt.subplots(2,5)
    for i, axi in enumerate(ax.flat):
        axi.imshow(np.reshape(w_matrix[i], (28, 28)), interpolation='nearest')
        axi.set_title("W" + str(i))
    #plt.show()


# best_ce_eta_classifier()

# (c)
def best_ce_classifier():
    train_data, train_labels, validation_data, validation_labels, test_data, test_labels = helper_ce()
    w_matrix = SGD_ce(train_data, train_labels, pow(10, -2), 20000)
    accuracy = accuracy_precentage_v2(w_matrix, test_data, test_labels)
    # print(accuracy)


# best_ce_classifier()

#################################

# Place for additional code

#################################
