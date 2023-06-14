import os
import time
import functools

import numpy as np

from config import RNN_RESULT_DIR, VOCAB_DIR
from data.utils import load_npy, load_pickle
from Fidelity import PrefixTree4Fidelity


# --------------------------------------- Pipeline ---------------------------------------

def read_results(name, model, read_train, read_valid, read_test):
    """ Helper function for reading RNN results."""
    result_dir = os.path.join(RNN_RESULT_DIR, name, model)

    result, loaded, vocab = {}, [], load_pickle(VOCAB_DIR, name)

    if read_train:
        loaded.append(load_npy(result_dir, 'train_data').item())
    if read_valid:
        try:
            loaded.append(load_npy(result_dir, 'valid_data').item())
        except FileNotFoundError:
            pass
    if read_test:
        loaded.append(load_npy(result_dir, 'test_data').item())

    for res in loaded:
        for i in ['input', 'hidden', 'output']:
            result[i] = np.concatenate((result.get(i, np.ndarray(
                (0, *res[i].shape[1:]), dtype=res[i].dtype)), res[i]), axis=0)

    return vocab, result['input'], result['hidden'], result['output']


class RNNLoader:

    def __init__(self, name, model, read_train=True, read_valid=True, read_test=True):
        """ Result loader of trained RNN for extracting patterns and building DFA.

        Args:
            name: str, choices: ['synthetic1', 'synthetic2', 'tomita1', 'tomita2', 'yelp']
            model: str: choices: ['rnn', 'lstm', 'gru']
            read_train: bool, whether to read train data
            read_valid: bool, whether to read validation data
            read_test: bool, whether to read test data

        Attributes:
            - vocab: RNN vocabulary for decoding sequence
            - alphabet: list of shape (VOCAB_SIZE)
            - rnn_data: (input_sequence, hidden_states, rnn_output)
                - input_sequence, np.array of shape (N, PAD_LEN)
                - hidden_states, np.array of shape (N, PAD_LEN, hidden_dim)
                - rnn_output, np.array of shape (N,)
                - decoded_input_seq, np.array of shape (N, PAD_LEN, hidden_dim), decoded input strings
            - prefix-tree: tree structure representation of input sequences, used for
                obtaining hidden values and calculating fidelity
        """

        vocab, *rnn_data = read_results(name, model, read_train, read_valid, read_test)
        self.vocab, self.alphabet = vocab, vocab.words
        self.input_sequences, self.hidden_states, self.rnn_prob_output = rnn_data
        self.rnn_output = self.rnn_prob_output.round()
        self.decoded_input_seq = np.array([self.decode(
            seq, remove_padding=False, as_list=True) for seq in self.input_sequences])
        self.prefix_tree = PrefixTree4Fidelity(self.decoded_input_seq, self.hidden_states, self.rnn_output)

    def decode(self, seq, remove_padding=True, as_list=False, sep=' '):
        text = [self.vocab.words[i] for i in seq]
        if remove_padding:
            text = [word for word in text if word != '<pad>']
        if as_list:
            return text
        return sep.join(text)

    def eval_fidelity(self, dfa):
        """ Evaluate the fidelity of (extracted) DFA from rnn_loader.

        Rather slow. Should only be used for test."""
        return np.mean([dfa.classify_expression(self.decode(expr, as_list=True)) == ro
                        for expr, ro in zip(self.input_sequences, self.rnn_output)])


# ---------------------------- math --------------------------------

def d(hidden1: np.array, hidden2: np.array):
    """ Euclidean distance of hidden state values."""
    return np.sqrt(np.sum((hidden1 - hidden2) ** 2))


# ------------------------- graphviz -------------------------------

def add_nodes(graph, nodes):
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


# ---------------------------- logger & timer ---------------------------------

def logger(func):
    """ Helper class for a logging decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"----- {func.__name__}: start -----")
        output = func(*args, **kwargs)
        print(f"----- {func.__name__}: end -----")
        return output

    return wrapper


def timeit(func):
    """ Helper function for a timing decorator."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        if end - start > 3:
            pass
        print(f'{func.__name__} took {end - start:.6f} seconds to complete')
        return result

    return wrapper


# ------------------------------- tools ---------------------------------------

class LazyAttribute(object):
    """ Helper class for lazy initializing class property."""

    def __init__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter = func

    def __get__(self, obj, cls):
        value = self.getter(cls)
        setattr(cls, self.__name__, value)
        return value


class ConfigDict:
    """ Helper class for configuration."""

    def __init__(self, config_dict: dict):
        # config_dict: a dict object holding configurations
        self.config = config_dict

    def __getattr__(self, item):
        return self.config[item]

    def update(self, new_config):
        self.config.update(new_config)


if __name__ == "__main__":
    loader = RNNLoader('tomita_data_1', 'lstm')
