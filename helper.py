
import numpy as np


def load_rnn_results():
    pass


# Euclidean distance of hidden state values
def d(hidden1, hidden2):
    return np.sqrt(np.sum((hidden1 - hidden2) ** 2))
