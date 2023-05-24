import os

import numpy as np

from config import START_SYMBOL, RNN_RESULT_DIR
from data.utils import load_npy


def read_results(fname, model):
    result_dir = os.path.join(RNN_RESULT_DIR, fname, model)

    train_result = load_npy(result_dir, 'train_data').item()
    test_result = load_npy(result_dir, 'test_data').item()

    try:
        valid_result = load_npy(result_dir, 'valid_data').item()
    except FileNotFoundError:
        valid_result = None

    result = {}
    for res in [train_result, valid_result, test_result]:
        for i in ['input', 'hidden', 'output']:
            if res:
                result[i] = np.concatenate((result.get(i, np.ndarray((0, *res[i].shape[1:]))), res[i]), axis=0)

    return res['input'], res['hidden'], ['output']


class RNNLoader:

    def __int__(self, alphabet, rnn_data):
        """ Result loader of trained RNN for extracting patterns and building DFA.

        Args:
            - alphabet: list of shape (VOCAB_SIZE)
            - rnn_data: (input_sequence, hidden_states, rnn_output)
                - input_sequence, np.array of shape (N, PAD_LEN)
                - hidden_states, np.array of shape (N, PAD_LEN, hidden_dim)
                - rnn_output, np.array of shape (N,)
        """
        self.alphabet = alphabet
        self.input_sequence, self.hidden_states, self.rnn_output = rnn_data

        # Check shape
        assert self.input_sequence.shape[0] == self.hidden_states.shape[0]
        assert self.input_sequence.shape[1] == self.hidden_states.shape[1]
        assert self.input_sequence.shape[0] == self.rnn_output.shape[0]

        # Check START_SYMBOL
        if START_SYMBOL:
            assert self.input_sequence[:, 0] == START_SYMBOL

    # The hidden value in the RNN for given prefix
    # todo: Accelerate by using cashed hidden states
    def rnn_hidden_values(self, prefix):
        pass


def d(hidden1: np.array, hidden2: np.array):
    """ Euclidean distance of hidden state values."""
    return np.sqrt(np.sum((hidden1 - hidden2) ** 2))


# todo: add logger


def add_nodes(graph, nodes):  # stolen from http://matthiaseisen.com/articles/graphviz/
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph


if __name__ == "__main__":
    pass
