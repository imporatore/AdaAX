# Ambiguity here.
# Hidden in core sets discovered during backtracking shares common suffix (and cluster label), not necessary the prefix.
# Hidden in core sets generated along 'extracted pattern', however, share common prefix, thence consistent hidden value.
# Core sets generated along 'extracted pattern' are called pure sets instead for disambiguation.

from collections import defaultdict

import numpy as np

from config import START_PREFIX


class PureSet:
    """ Pure Sets are determined by only one prefix(path), so have consistent hidden value."""

    def __init__(self, hidden_eval_func, prefix):
        """

        Args:
            hidden_eval_func
            prefix: list, a list of symbols which form the path from the start h0 to this 'pure set'.
        """
        self._prefix = prefix
        self._h = hidden_eval_func(prefix) if prefix else None  # hidden value

    # def __repr__(self):
    #     pass


class State(PureSet):
    """ State(in DFA) is a set of 'PureSet'(prefixes).

    Holds the prefixes of all PureSets this state contains."""

    def __init__(self, hidden_eval_func, prefix):
        """

        Args:
            hidden_eval_func
            prefix: list, a list of symbols which initialize the State (as a PureSet).
                - The first prefix upon which the PureSet is built.
                - Also, the hidden value of this PureSet is evaluated on this prefix.
            prev_state: the previous state which transit into this State by symbol prefix[-1],
                - also the state which represent prefix prefix[:-1].
        """
        # todo: hidden state value is set to be a constant and never updates, even after merging.
        super().__init__(hidden_eval_func, prefix)

        self.parents = defaultdict(list)  # parent set dict: {symbol: [prev_states]} prev_state ---symbol--> state

        self._eval_hidden_value = hidden_eval_func  # hidden value function
        # todo: add support for each prefix

    # # CANNOT USE DUE TO LOOP
    # # todo: prefixes can be updated explicitly in the consolidation stage
    # # However, as long as it was seldom called, it can remain here.
    # @property
    # def prefixes(self):
    #     """ All prefixes which goes to the state.
    #
    #     Returns:
    #         prefixes: list[list], list of prefixes, where each prefix is a list of symbols
    #     """
    #     if self._prefix == START_PREFIX:  # start state
    #         return [START_PREFIX]
    #     res = []
    #     for s, states in self.parents.items():
    #         for state in states:
    #             res.extend([prefix + s for prefix in state.prefixes])
    #     return res
    #
    # @property
    # def h(self):
    #     """ Evaluate the new hidden value (after merged). Unused for now."""
    #     # todo: weighted (by support) average of all prefixes
    #     hidden_vals = [self._eval_hidden_value(p) for p in self.prefixes]
    #     return np.mean(np.array(hidden_vals), axis=0)

    def __copy__(self):
        """ Shallow copy of states which pointer to the parents list changes but pointer to the parents sets remains"""
        prefix, eval_func = self._prefix, self._eval_hidden_value
        new_copy = State(eval_func, prefix)
        if prefix is None:  # accept state
            new_copy._h = self._h
        for s in self.parents.keys():
            # a shallow copy for the list of states for parents of given symbol s
            new_copy.parents[s] = self.parents[s].copy()
            # this step here would cause a bug in copying dfa (when updating parents at the same time, self loop is
            # already updated so a KeyError would occur), so deprecated here
            # todo: better update all parent and delta when a new state is generated.
            # change parent if self loop
            # if self in new_copy.parents[s]:
            #     new_copy.parents[s].remove(self)
            #     new_copy.parents[s].append(new_copy)
        return new_copy

    # def __eq__(self, other):  # if added, cannot use hash dict
    #     return self.parents == other.parents  # todo: should empty list be removed? do they exist?


def build_start_state(hidden_eval_func):
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
    return State(hidden_eval_func, START_PREFIX)


def build_accept_state(rnn_loader):
    """ Build accept state for the DFA.

    Args:
        rnn_loader

    Returns:
        h0, a 'PureSet'(State) of start state.
    """
    F = State(rnn_loader.rnn_hidden_values, None)

    # Accept states owns all positive prefixes (added in add_pattern func)
    # todo: it is likely that they aren't used, so can be duplicated
    # F._prefix = rnn_loader.input_sequence[rnn_loader.rnn_output == 1]
    F._h = np.mean(rnn_loader.hidden_states[rnn_loader.rnn_output == 1, -1, :], axis=0)
    return F
