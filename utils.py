import os
import time
import functools

import numpy as np

from config import START_SYMBOL, START_PREFIX, RNN_RESULT_DIR, VOCAB_DIR
from data.utils import load_npy, load_pickle


# ----------------------------------- Data Structure ----------------------------------------

class SymbolNode(object):

    def __init__(self, val):
        self.val = val  # Symbol in the alphabet
        self.next = []

    def __getattr__(self, item):
        if item in ['pos_sup', 'neg_sup']:
            try:
                return super().__getattr__(item)
                # return self.__dict__[item]
            except AttributeError:
                # except KeyError:
                return 0
        return super().__getattr__(item)
        # return self.__dict__[item]

    @property
    def sup(self):
        if self.pos_sup or self.neg_sup:
            return self.pos_sup - self.neg_sup
        raise AttributeError("Node has neither positive support nor negative support.")


class PrefixTree:
    def __init__(self, seq, hidden):
        self.root = SymbolNode(START_SYMBOL)
        self.root.h = hidden[0, 0, :]  # hidden values for start symbol
        self._build_tree(seq, hidden)

    def _build_tree(self, seq, hidden):
        for s, h in zip(seq, hidden):
            self._update(s[len(START_PREFIX):], h[len(START_PREFIX):])

    def _update(self, s, h):
        cur = self.root
        for i, symbol in enumerate(s):
            for n in cur.next:
                if n.val == symbol:
                    cur = n
                    break
            else:
                node = SymbolNode(symbol)
                node.h = h[i, :]
                cur.next.append(node)
                cur = node

    def eval_hidden(self, expr):
        cur = self.root
        for symbol in expr:
            for n in cur.next:
                if n.val == symbol:
                    cur = n
        return cur.h


# --------------------------------------- Pipeline ---------------------------------------

def read_results(name, model):
    result_dir = os.path.join(RNN_RESULT_DIR, name, model)

    train_result = load_npy(result_dir, 'train_data').item()
    test_result = load_npy(result_dir, 'test_data').item()

    try:
        valid_result = load_npy(result_dir, 'valid_data').item()
    except FileNotFoundError:
        valid_result = None

    vocab = load_pickle(VOCAB_DIR, name)

    result = {}
    for res in [train_result, valid_result, test_result]:
        if res:
            for i in ['input', 'hidden', 'output']:
                result[i] = np.concatenate((result.get(i, np.ndarray(
                    (0, *res[i].shape[1:]), dtype=res[i].dtype)), res[i]), axis=0)

    return vocab, result['input'], result['hidden'], result['output']


class RNNLoader:

    def __init__(self, name, model):
        """ Result loader of trained RNN for extracting patterns and building DFA.

        Attributes:
            - alphabet: list of shape (VOCAB_SIZE)
            - rnn_data: (input_sequence, hidden_states, rnn_output)
                - input_sequence, np.array of shape (N, PAD_LEN)
                - hidden_states, np.array of shape (N, PAD_LEN, hidden_dim)
                - rnn_output, np.array of shape (N,)
        """

        vocab, *rnn_data = read_results(name, model)
        self.vocab, self.alphabet = vocab, vocab.words
        self.input_sequences, self.hidden_states, self.rnn_prob_output = rnn_data
        self.rnn_output = self.rnn_prob_output.round()
        self.decoded_input_seq = np.array([self.decode(
            seq, remove_padding=False, as_list=True) for seq in self.input_sequences])
        self.prefix_tree = PrefixTree(self.decoded_input_seq, self.hidden_states)

    # The hidden value in the RNN for given prefix
    # todo: Accelerate by using cashed hidden states
    def rnn_hidden_values(self, prefix):
        if START_SYMBOL:
            return self.prefix_tree.eval_hidden(prefix[1:])
        return self.prefix_tree.eval_hidden(prefix)

    def decode(self, seq, remove_padding=True, as_list=False, sep=' '):
        text = [self.vocab.words[i] for i in seq]
        if remove_padding:
            text = [word for word in text if word != '<pad>']
        if as_list:
            return text
        return sep.join(text)

    # todo: only called when merging, may use cashed result to accelerate, as the difference in the merged DFA is small
    def fidelity(self, dfa):
        """ Evaluate the fidelity of (extracted) DFA from rnn_loader."""
        return np.mean([dfa.classify_expression(self.decode(expr, as_list=True)) == ro for expr, ro in zip(
            self.input_sequences, self.rnn_output)])


# ---------------------------- math --------------------------------

def d(hidden1: np.array, hidden2: np.array):
    """ Euclidean distance of hidden state values."""
    return np.sqrt(np.sum((hidden1 - hidden2) ** 2))


# ------------------------- plotting -------------------------------

def add_nodes(graph, nodes):  # stolen from http://matthiaseisen.com/articles/graphviz/
    for n in nodes:
        if isinstance(n, tuple):
            graph.node(n[0], **n[1])
        else:
            graph.node(n)
    return graph


def add_edges(graph, edges):  # stolen from http://matthiaseisen.com/articles/graphviz/
    for e in edges:
        if isinstance(e[0], tuple):
            graph.edge(*e[0], **e[1])
        else:
            graph.edge(*e)
    return graph


# ---------------------------- logger ---------------------------------

def logger(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f"----- {func.__name__}: start -----")
        output = func(*args, **kwargs)
        print(f"----- {func.__name__}: end -----")
        return output

    return wrapper


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f'{func.__name__} took {end - start:.6f} seconds to complete')
        return result

    return wrapper


# ------------------------------- tools ---------------------------------------

class LazyAttribute(object):
    """ A property that caches itself to the class object. """

    def __init__(self, func):
        functools.update_wrapper(self, func, updated=[])
        self.getter = func

    def __get__(self, obj, cls):
        value = self.getter(cls)
        setattr(cls, self.__name__, value)
        return value


if __name__ == "__main__":
    loader = RNNLoader('tomita_data_1', 'lstm')
