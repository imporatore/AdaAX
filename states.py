# Ambiguity here.
# Hidden in core sets discovered during backtracking shares common suffix (and cluster label), not necessary the prefix.
# Hidden in core sets generated along 'extracted pattern', however, share common prefix, thence consistent hidden value.
# Core sets generated along 'extracted pattern' are called pure sets instead for disambiguation.

from collections import defaultdict

import numpy as np

from config import START_SYMBOL


class PureSet:
    """ Pure Sets are determined by only one prefix(path), so have consistent hidden value."""

    def __init__(self, rnn_loader, prefix: list):
        """

        Args:
            - rnn_loader
            - prefix: list, a list of symbols which form the path from the start h0 to this 'pure set'.
        """
        self._prefix = prefix
        self._h = rnn_loader.rnn_hidden_values(prefix)  # hidden value


class State:
    """ State(in DFA) is a set of 'PureSet'(prefixes).

    Holds the prefixes of all PureSets this state contains."""

    def __init__(self, rnn_loader, prefix: list):
        """

        Args:
            - rnn_loader
            - prefix: list, a list of symbols which initialize the State (as a PureSet).
        """
        # todo: hidden state value is set to be a constant and never updates, even after merging.
        self._prefix = [prefix]
        self._h = rnn_loader.rnn_hidden_values(prefix)  # hidden value
        self._rnn_hidden_value = rnn_loader.rnn_hidden_values  # hidden value function

    @property
    def prefixes(self):
        return self._prefix

    @property
    def initial_prefix(self):
        """ The first prefix upon which the PureSet is built.

        Also, the hidden value of this PureSet is evaluated on this prefix.
        """
        return self._prefix[0]

    @property
    def h(self):
        """ Evaluate the new hidden value (after merged). Unused for now."""
        # todo: weighted (by support) average of all prefix
        hidden_vals = [self._rnn_hidden_value(p) for p in self._prefix]
        return np.mean(np.array(hidden_vals), axis=0)


def build_start_state(rnn_loader):
    """ Build start state for the DFA.

    Args:
        - rnn_loader

    Params:
        - START_SYMBOL: symbol for represent sentence start,
            added to all input sequence for a uniform start sign and hidden value.
            - str, symbol of expression start, should be added to RNN training (thus alphabet, input sequence & hidden)
            - todo: None, if no start sign is added

    Returns:
        - h0, a 'PureSet'(State) of start state.
    """
    return State(rnn_loader, [START_SYMBOL])


def build_accept_state(rnn_loader):
    """ Build accept state for the DFA.

    Args:
        - rnn_loader

    Returns:
        - h0, a 'PureSet'(State) of start state.
    """
    F = State(rnn_loader, [None])

    # Accept states owns all positive prefixes
    # todo: it is likely that they aren't used, so can be duplicated
    F._prefix = rnn_loader.input_sequence[rnn_loader.rnn_output == 1]
    F._h = np.mean([rnn_loader.rnn_hidden_values(p) for p in F.prefixes], axis=0)
    return F
