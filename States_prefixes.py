# Ambiguity here.
# Hidden in core sets discovered during backtracking shares common suffix (and cluster label), not necessary the prefix.
# Hidden in core sets generated along 'extracted pattern', however, share common prefix, thence consistent hidden value.
# Core sets generated along 'extracted pattern' are called pure sets instead for disambiguation.

import numpy as np

from config import START_PREFIX


class PureSet:
    """ Pure Sets are determined by only one prefix(path), so have consistent hidden value."""

    def __init__(self, prefix):
        """

        Args:
            prefix: list, a list of symbols which form the path from the start h0 to this 'pure set'.
        """
        self._prefix = prefix

    def __repr__(self):
        pass


class State:
    """ State(in DFA) is a set of 'PureSet'(prefixes).

    Holds the prefixes of all PureSets this state contains."""

    def __init__(self, prefixes):
        """

        Args:
            prefix: list, a list of symbols which initialize the State (as a PureSet).
                - The first prefix upon which the PureSet is built.
                - Also, the hidden value of this PureSet is evaluated on this prefix.
            prev_state: the previous state which transit into this State by symbol prefix[-1],
                - also the state which represent prefix prefix[:-1].
        """
        # todo: hidden state value is set to be a constant and never updates, even after merging.
        if not prefixes:
            self.prefixes = []
        elif not isinstance(prefixes, list):
            raise ValueError("Argument prefixes must be a list.")
        # todo: add support for each prefix
        elif not isinstance(prefixes[0], list):
            self.prefixes = [prefixes]
        else:
            self.prefixes = prefixes

    def h(self, rnn_loader):
        """ Evaluate the new hidden value (after merged). Unused for now."""
        # todo: weighted (by support) average of all prefixes
        hidden_vals = [rnn_loader.rnn_hidden_values(p) for p in self.prefixes]
        return np.mean(np.array(hidden_vals), axis=0)


def build_start_state():
    """ Build start state for the DFA.

    Args:
        hidden_eval_func

    Params:
        START_SYMBOL: symbol for represent sentence start,
            added to all input sequence for a uniform start sign and hidden value.
            - str, symbol of expression start, should be added to RNN training (thus alphabet, input sequence & hidden)
            - None, if no start sign is added

    Returns:
        h0, a 'PureSet'(State) of start state.
    """
    return State(START_PREFIX)


def build_accept_state():
    """ Build accept state for the DFA.

    Returns:
        h0, a 'PureSet'(State) of start state.
    """
    F = State(None)

    # Accept states owns all positive prefixes (added in add_pattern func)
    # todo: it is likely that they aren't used, so can be duplicated

    # F._h = np.mean(rnn_loader.hidden_states[rnn_loader.rnn_output == 1, -1, :], axis=0)
    return F


if __name__ == "__main__":
    from utils import RNNLoader

    loader = RNNLoader('tomita_data_1', 'lstm')
    start_state = build_start_state()
    print(start_state.h(loader))
