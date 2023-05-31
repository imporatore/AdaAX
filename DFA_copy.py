import copy
import functools
from collections import defaultdict

import numpy as np
import graphviz as gv
from IPython.display import Image
from IPython.display import display

from States_copy import State, build_start_state, build_accept_state
from utils import d, add_nodes, add_edges, LazyAttribute
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
        self.q0 = build_start_state(rnn_loader.rnn_hidden_values)  # start state
        self.F = build_accept_state(rnn_loader)  # accept state
        self.Q = [self.q0, self.F]  # states
        self.delta = defaultdict(dict)  # transition table

        self._rnn_loader = rnn_loader

    # todo: add pattern by PatternTree
    # todo: require testing
    # todo: if the merging stage is cooperated in the adding stage, would it be faster?
    def _add_pattern(self, p):
        """ Add new? pattern to DFA

        Args:
            p: list, pattern is a list of symbols

        Returns:
            Q_new: list, new states (pure sets) added by the pattern
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
            patterns: list[list], list of patterns

        Params:
            TAU: threshold for neighbour distance
            DELTA: threshold for merging fidelity loss
        """
        for p in patterns:
            # list of new states created by pattern
            A_t = self._add_pattern(p[len(START_PREFIX):])  # if START_SYMBOL, first symbol in pattern is START_SYMBOL
            while A_t:

                assert all([st in self.Q for st in A_t])

                # try merge new states in A_t
                q_t = A_t.pop()
                N_t = {s: d(q_t._h, s._h) for s in self.Q if s != q_t}  # neighbours of q_t
                # N_t = {s: d(q_t._h, s._h) for s in self.Q if s not in (q_t, self.F)}
                # N_t = {s: d(q_t._h, s._h) for s in self.Q if s not in (q_t, self.F, self.q0)}
                for s in sorted(N_t.keys(), key=lambda x: N_t[x]):
                    if N_t[s] >= TAU:  # threshold (Euclidean distance of hidden values) for merging states
                        break

                    new_dfa, mapping = self._merge_states(q_t, s)  # create the DFA after merging
                    if self.fidelity - new_dfa.fidelity < DELTA:  # accept merging if fidelity loss below threshold
                        self.Q, self.q0, self.F = new_dfa.Q, new_dfa.q0, new_dfa.F  # update states
                        self.delta = new_dfa.delta  # update transitions
                        A_t = [mapping[state] for state in A_t if mapping[state]]
                        break

    def __copy__(self):
        """ Shallow copy for merging"""
        # todo: part unchanged， part changed
        new_dfa = DFA(self._rnn_loader)
        new_dfa.q0 = copy.copy(self.q0)
        new_dfa.F = copy.copy(self.F)
        new_dfa.Q = [new_dfa.q0, new_dfa.F]
        new_dfa.delta = defaultdict(dict)
        mapping = defaultdict()
        mapping.update({self.q0: new_dfa.q0, self.F: new_dfa.F})

        for state in self.Q:
            if state not in [self.q0, self.F]:
                new_state = copy.copy(state)
                new_dfa.Q.append(new_state)
                mapping.update({state: new_state})

        for state in new_dfa.Q:
            state.parents.update({s: [mapping[p] for p in state.parents[s]] for s in state.parents.keys()})

        for state in self.delta.keys():
            new_dfa.delta[mapping[state]] = {s: mapping[self.delta[state][s]] for s in self.delta[state].keys()}

        return new_dfa, mapping

    # todo: require testing
    # todo: test for self-loop and transition to the state it merges with
    def _merge_states(self, state1, state2):
        """ Try merging state1 with state2.

        Notice that if the child state not consistent, they will also be merged.

        Args:
            state1:
            state2:

        Returns:
            new_dfa: new DFA after merging state1 with state2 in the existing DFA
        """
        # todo: forbid merging accept state
        # todo: add threshold for merging accept state
        # todo: the hidden state values remains after merging
        new_dfa, mapping = copy.copy(self)
        mapped_state1, mapped_state2 = mapping[state1], mapping[state2]

        if state1 == self.q0:
            new_dfa.q0 = mapped_state2
        if state1 == self.F:
            new_dfa.F = mapped_state2

        for state in new_dfa.Q:
            for s in state.parents.keys():
                if mapped_state1 in state.parents[s]:
                    state.parents[s].remove(mapped_state1)
                    if mapped_state2 not in state.parents[s]:
                        state.parents[s].append(mapped_state2)

        for s in mapped_state1.parents.keys():
            for parent in mapped_state1.parents[s]:
                if parent not in mapped_state2.parents[s]:
                    mapped_state2.parents[s].append(parent)
                    new_dfa.delta[parent][s] = mapped_state2

        mapping[state1] = None
        new_dfa.Q.remove(mapped_state1)  # update states

        # todo: seems no self loop with mapped_state1 & mapped_state2 exists
        if mapped_state1 in new_dfa.delta.keys():
            # Update outgoing transitions
            transition = new_dfa.delta.pop(mapped_state1)

            for s in transition.keys():
                child = transition[s] if transition[s] != mapped_state1 else mapped_state2
                if s not in new_dfa.delta[mapped_state2].keys():
                    new_dfa.delta[mapped_state2][s] = child
                else:
                    child_ = new_dfa.delta[mapped_state2][s] if new_dfa.delta[mapped_state2][s] != mapped_state1 else mapped_state2
                    if child_ == child:
                        new_dfa.delta[mapped_state2][s] = child
                    else:
                        new_dfa, mapping_ = new_dfa._merge_states(child, child_)
                        mapping.update({st: mapping_[mapping[st]] if mapping[st] else None for st in mapping.keys()})
                        # new_dfa = new_dfa._merge_states(child, child_)
                        # new_dfa.__mapping = {st: new_dfa.__mapping[mapping[st]] for st in mapping.keys() if mapping[st]}

        for st in new_dfa.Q:
            for s in st.parents.keys():
                assert all([p in new_dfa.Q for p in st.parents[s]])

        assert all([st in new_dfa.Q for st in new_dfa.delta.keys()])

        for st in new_dfa.delta.keys():
            assert all([c in new_dfa.Q for c in new_dfa.delta[st].values()])

        return new_dfa, mapping

    def prefix2state(self, prefix):
        """ Return the state in DFA for prefix."""
        if prefix == START_PREFIX:
            return self.q0
        return self.delta[self.prefix2state(prefix[:-1])][prefix[-1]]

    def _add_new_state(self, prefix, prev_state):
        """ Add and return the new state from a new prefix."""
        state = State(self._rnn_loader.rnn_hidden_values, prefix)  # Initialize pure set from new prefix

        # Update DFA
        self.Q.append(state)  # Add to states
        self._add_transit(prev_state, prefix[-1], state)  # Update transition
        return state

    # todo: as we only extract patterns from positive samples, and DFA entirely built on these patterns
    # todo: how to deal with missing transitions?
    # todo: needs modification
    def classify_expression(self, expression):
        # Expression is string with only letters in alphabet
        q = self.q0
        for s in expression[len(START_PREFIX):]:
            try:
                q = self.delta[q][s]
            except KeyError:  # if no transition found, then expression is not among the extracted patterns
                return False
        # return q == self.F
        return True

    # todo: only called when merging, may use cashed result to accelerate, as the difference in the merged DFA is small
    @LazyAttribute
    @property
    def fidelity(self):
        """ Evaluate the fidelity of (extracted) DFA from rnn_loader."""
        return np.mean([self.classify_expression(self._rnn_loader.decode(expr, as_list=True)) == ro for expr, ro in zip(
            self._rnn_loader.input_sequences, self._rnn_loader.rnn_output)])

    def _add_transit(self, state1, symbol, state2):
        """ Add a transition from state1 to state2 by symbol."""
        self.delta[state1][symbol] = state2  # update transition table
        if state1 not in state2.parents[symbol]:
            state2.parents[symbol].append(state1)  # update parent set of state2

    @LazyAttribute
    @property
    def _edges(self):
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

        g = add_edges(g, [(e, {'label': SEP.join(self._edges[e])}) for e in self._edges.keys()])

        display(Image(filename=g.render(filename='img/automaton')))


if __name__ == "__main__":
    from utils import RNNLoader
    from Pattern import pattern_extraction

    loader = RNNLoader('tomita_data_1', 'lstm')
    patterns, supports = pattern_extraction(loader)
    dfa = DFA(loader)
    dfa.build_dfa(patterns)

    pass
