from collections import Counter

import numpy as np

from DFA import DFA



class AdaAX:

    def __init__(self):
        self._alphabet = None
        self._pattern = []
        self._dfa = DFA()

    def _load_data(self, alphabet, sentence, hidden, labels):
        """

        Params:

            - alphabet: list, words used in the neural net
                - should use one-hot encoding, or,
                - is word embeddings acceptable?
            - sentence: np.array, shape (n_samples, len_pad/truncate)
                - best of


        :param hidden:
        :param labels:
        :return:
        """
        self._alphabet = alphabet

    def _extract_pattern(self, ):


PAT = []  # Extracted pattern
START = '<START>'  # Start symbol
# Cluster_id: 0, 1, 2, ... , C-1
CLUSTERS = 10  # Initial cluster numbers, determined by elbow method ########## to-do auto
PRUNE_THRESHOLD = 0.005  # Threshold for pruning focal set

def extract_pattern(hidden_states, pattern):
    """ Extract patterns by DFS backtracking at the level of focal sets.

    Params:

        - hidden_states: list, hidden states of a focal set.
        - pattern: list

    Return:
        Update extracted pattern.
    """
    # Reaches start state and add pattern
    # hidden_state[0] == START
    # todo: pattern support
    if all([h == START for h in hidden_states]):
        PAT.append(pattern)
        return

    # Split previous states by cluster_ids
    # ! Could be moved inside the for loop of cluster to save memory
    prev_hidden_states = [[h._prev for h in hidden_states if h._prev._cluster == c] for c in range(CLUSTERS)]
    # Sort cluster ids by the size each sub cluster
    cluster_count = [len(phs) for phs in prev_hidden_states]
    sorted_cluster_ids = sorted(range(CLUSTERS), key=lambda x: cluster_count[x], reverse=True)

    for c in sorted_cluster_ids:
        # Prune if the size of sub cluster is too small
        # ? Actually we are not calculating the data support of core set, instead we sum by symbols
        # ! Use break instead of continue since we have already sorted cluster_ids by its size in descent order
        if cluster_count[c] < PRUNE_THRESHOLD:
            break

        # Symbols backtracked from hidden_states
        # ! Moved it inside the loop since the symbols used by sub clusters may not correspond to the whole prev states
        # symbols = set([h._next_symbol for h in prev_hidden_states[c]])

        # ! Likewise, move it outside the for loop for accelerating
        prev_hidden_states_by_symbol = {}
        for h in prev_hidden_states[c]:
            prev_hidden_states_by_symbol[h._next_symbol] = prev_hidden_states_by_symbol.get(h._next_symbol, []) + [h]
        # sorted_symbols = sorted(symbols, key=lambda x: prev_hidden_states_by_symbol[x])

        # Search each symbol in descent order
        # ? Prune trivial symbols?
        for s in sorted(prev_hidden_states_by_symbol.keys(), key=lambda x: prev_hidden_states_by_symbol[x], reverse=True):
            # prepend symbol to the current pattern
            pat = [s] + pattern
            extract_pattern(prev_hidden_states_by_symbol[s], pat)

def add_pattern(pattern):
    cursor = START

if __name__ == "__main__":
    from config import RNN_HIDDEN_DIR, LSTM_HIDDEN_DIR, GRU_HIDDEN_DIR

