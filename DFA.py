import copy
import functools
from collections import defaultdict

import numpy as np
import graphviz as gv
from IPython.display import Image
from IPython.display import display

from States import State, build_start_state, build_accept_state
from utils import d, add_nodes, add_edges
from config import TAU, DELTA, START_PREFIX, SEP

digraph = functools.partial(gv.Digraph, format='png')
graph = functools.partial(gv.Graph, format='png')


# todo: add state split to ensure gradually learning more & more difficult patterns from flow or samplers
class DFA:
    """
    Attributes:

    """
    def __init__(self, rnn_loader):
        self.alphabet = rnn_loader.alphabet  # alphabet
        self.q0 = build_start_state(rnn_loader)  # start state
        self.F = build_accept_state(rnn_loader)  # accept state
        self.Q = [self.q0, self.F]  # states
        self.delta = defaultdict(dict)  # transition table

        # todo: check if this would take a lot of memory
        self._rnn_loader = rnn_loader

    # todo: add pattern by PatternTree
    # todo: require testing
    # todo: if the merging stage is cooperated in the adding stage, would it be faster?
    def _add_pattern(self, p):
        """ Add new? pattern to DFA

        Args:
            - p: list, pattern is a list of symbols

        Returns:
            - Q_new: list, new states (pure sets) added by the pattern
        """
        q1 = self.q0
        Q_new = []  # New pure sets to add

        # new state shouldn't be created for the last symbol in a pattern, since it is the accept state
        for i, s in enumerate(p[:-1]):
            if s in self.delta[q1].keys():
                q1 = self.delta[q1][s]
            else:
                q1 = self._add_new_state(START_PREFIX + p[:i + 1], q1)
                Q_new.append(q1)

        self._add_transit(q1, p[-1], self.F)  # add transition of the last symbol to accept state
        return Q_new

    # todo: add pattern by PatternTree
    # todo: require testing
    def build_dfa(self, patterns):
        """ Build DFA using extracted patterns

        Args:
            - patterns: list[list], list of patterns

        Params:
            - TAU: threshold for neighbour distance
            - DELTA: threshold for merging fidelity loss
        """
        for p in patterns:
            # list of new states created by pattern
            A_t = self._add_pattern(p[len(START_PREFIX):])  # if START_SYMBOL, first symbol in pattern is START_SYMBOL
            while A_t:

                # try merge new states in A_t
                q_t = A_t.pop()
                N_t = {s: d(q_t._h, s._h) for s in self.Q if s != q_t}  # neighbours of q_t
                # N_t = {s: d(q_t._h, s._h) for s in self.Q if s not in (q_t, self.F)}
                # N_t = {s: d(q_t._h, s._h) for s in self.Q if s not in (q_t, self.F, self.q0)}
                for s in sorted(N_t.keys(), key=lambda x: N_t[x]):
                    if N_t[s] >= TAU:  # threshold (Euclidean distance of hidden values) for merging states
                        break

                    new_dfa = self._merge_states(q_t, s)  # create the DFA after merging
                    if self.fidelity - new_dfa.fidelity < DELTA:  # accept merging if fidelity loss below threshold
                        self.Q, self.q0, self.F = new_dfa.Q, new_dfa.q0, new_dfa.F  # update states
                        self.delta = new_dfa.delta  # update transitions
                        break

    # todo: require testing
    # todo: test for self-loop and transition to the state it merges with
    def _merge_states(self, state1, state2):
        """ Try merging state1 with state2.

        Notice that if the child state not consistent, they will also be merged.

        Args:
            - state1:
            - state2:

        Returns:
            - new_dfa: new DFA after merging state1 with state2 in the existing DFA
        """
        # todo: forbid merging accept state
        # todo: add threshold for merging accept state
        # todo: the hidden state values remains after merging
        new_dfa, new_state = copy.copy(self), copy.copy(state2)

        # todo: check if start & accept state would be merged
        # Update start and accept states if merged.
        if state2 == self.q0:
            new_dfa.q0 = new_state
        elif state2 == self.F:
            new_dfa.F = new_state

        # update children set
        for s in state1.parents.keys():
            new_state.parents[s] += state1.parents[s]

        # update states
        new_dfa.Q.remove(state1)
        new_dfa.Q.remove(state2)
        new_dfa.Q.append(new_state)

        # Update income transitions
        for s in new_state.parents.keys():
            for state in new_state.parents[s]:
                new_dfa.delta[state][s] = new_state

        # Update outgoing transitions
        transition1, transition2 = new_dfa.delta.pop(state1), new_dfa.delta.pop(state2)
        for s in transition1.keys():
            child1 = transition1.pop(s)
            try:
                child2 = transition2.pop(s)
                if child1 != child2:
                    # Merge outgoing states for common outgoing symbol if child state doesn't correspond
                    new_dfa = new_dfa._merge_states(child1, child2)
                else:
                    # update consistent child state
                    new_dfa._add_transit(new_state, s, child1)
                    child1.parents[s].remove(state1)
                    child2.parents[s].remove(state2)
            except KeyError:  # outgoing symbol only in state1
                new_dfa._add_transit(new_state, s, child1)
                child1.parents[s].remove(state1)

        for s, child in transition2.items():  # outgoing symbol only in state2
            new_dfa._add_transit(new_state, s, child)
            child.parents[s].remove(state2)

        return new_dfa

    def prefix2state(self, prefix):
        """ Return the state in DFA for prefix."""
        if prefix == START_PREFIX:
            return self.q0
        return self.delta[self.prefix2state(prefix[:-1])][prefix[-1]]

    def _add_new_state(self, prefix, prev_state):
        """ Add and return the new state from a new prefix."""
        state = State(self._rnn_loader, prefix, prev_state)  # Initialize pure set from new prefix

        # Update DFA
        self.Q.append(state)  # Add to states
        self._add_transit(prev_state, prefix[-1], state)  # Update transition
        return state

    # todo: require testing
    # todo: as we only extract patterns from positive samples, and DFA entirely built on these patterns
    # todo: how to deal with missing transitions?
    # todo: needs modification
    def classify_expression(self, expression):
        # Expression is string with only letters in alphabet
        q = self.q0
        for s in expression[len(START_PREFIX):]:
            q = self.delta[q][s]
        return q == self.F

    # todo: require testing
    # todo: only called when merging, may use cashed result to accelerate, as the difference in the merged DFA is small
    @property
    def fidelity(self):
        """ Evaluate the fidelity of (extracted) DFA from rnn_loader."""
        return np.mean([self.classify_expression(expr) == ro for expr, ro in zip(
            self._rnn_loader.input_sequence, self._rnn_loader.rnn_output)])

    def _add_transit(self, state1, symbol, state2):
        """ Add a transition from state1 to state2 by symbol."""
        self.delta[state1][symbol] = state2  # update transition table
        state2.parents[symbol] += [state1]  # update parent set of state2

    @property
    def edges(self):
        edges_dict = defaultdict(list)

        for parent in self.delta.keys():
            for s in self.delta[parent].keys():
                child = self.delta[parent][s]
                edges_dict[(parent, child)] += [s]

        return edges_dict

    # todo: require testing.
    def plot(self, force=False, maximum=60):

        if (not force) and len(self.Q) > maximum:
            raise Warning('State number exceeds limit (Maximum %d).' % maximum)

        state2int = {}

        def _state2int(state):
            if state not in state2int.keys():
                state2int[state] = max(state2int.values()) + 1
            return state2int[state]

        g = digraph()
        g = add_nodes(g, [(_state2int(self.q0), {'color': 'black', 'shape': 'hexagon', 'label': 'Start'})])
        g = add_nodes(g, [(_state2int(state), {'color': 'black', 'label': str(_state2int(state))})
                          for state in self.Q if state not in (self.q0, self.F)])
        g = add_nodes(g, [(_state2int(self.F), {'color': 'green', 'shape': 'hexagon', 'label': 'Accept'})])

        g = add_edges(g, [(e, {'label': SEP.join(self.edges[e])}) for e in self.edges.keys()])

        display(Image(filename=g.render(filename='img/automaton')))


if __name__ == "__main__":
    pass
