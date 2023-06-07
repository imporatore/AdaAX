import functools
from collections import defaultdict

import graphviz as gv
from IPython.display import Image
from IPython.display import display

from States_prefixes import State
from utils import add_nodes, add_edges
from config import START_PREFIX, SEP

digraph = functools.partial(gv.Digraph, format='png')
graph = functools.partial(gv.Graph, format='png')


# todo: add state split to ensure gradually learning more & more difficult patterns from flow or samplers
class DFA:
    """
    Attributes:

    """
    def __init__(self, alphabet, start_state, accept_state):
        self.alphabet = alphabet  # alphabet
        self.q0 = start_state  # start state
        self.F = accept_state  # accept state
        self.Q = [self.q0, self.F]  # states

        self.delta = defaultdict(dict)  # transition table

    # todo: use cashed result to accelerate
    def prefix2state(self, prefix):
        """ Return the state in DFA for prefix."""
        if prefix == START_PREFIX:
            return self.q0
        return self.delta[self.prefix2state(prefix[:-1])][prefix[-1]]

    def add_new_state(self, prefix, hidden, prev=None):
        """ Add and return the new state from a new prefix."""
        state = State(prefix)  # Initialize pure set from new prefix
        state._h = hidden

        # Update DFA
        self.Q.append(state)  # Add to states

        if prev:
            if isinstance(prev, list):
                for p in prev:
                    self.add_transit(p, prefix[-1], state)  # Update transition
            else:
                self.add_transit(prev, prefix[-1], state)  # Update transition

        return state

    def add_transit(self, state1, symbol, state2):
        """ Add a transition from state1 to state2 by symbol."""
        self.delta[state1][symbol] = state2  # update transition table

        # todo: use graph-like node link instead of prefix set
        # the prefixes cannot exceed the training set, or there will be problem concerning self loops and
        # infinite prefixes
        # for prefix in state1.prefixes:
        #     if prefix + [symbol] not in state2.prefixes:
        #         state2.prefixes.append(prefix + [symbol])

    # todo: require testing
    # todo: as we only extract patterns from positive samples, and DFA entirely built on these patterns
    # todo: how to deal with missing transitions?
    # todo: needs modification
    def classify_expression(self, expression):
        # Expression is string with only letters in alphabet
        q = self.q0
        for s in expression[len(START_PREFIX):]:
            q = self.delta[q][s]
            if not q:  # if no transition found, then expression is not among the extracted patterns
                return False
            if q == self.F:
                return True
        # return q == self.F
        # return True
        return False

    def fidelity(self, rnn_loader):
        return rnn_loader.eval_fidelity(self)

    # todo: require testing
    def plot(self, force=False, maximum=60):

        edges = defaultdict(list)

        for parent in self.delta.keys():
            for s in self.delta[parent].keys():
                child = self.delta[parent][s]
                edges[(parent, child)] += [s]

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

        g = add_edges(g, [(e, {'label': SEP.join(edges[e])}) for e in edges.keys()])

        display(Image(filename=g.render(filename='img/automaton')))


if __name__ == "__main__":
    from utils import RNNLoader

    loader = RNNLoader('tomita_data_1', 'lstm')
    dfa = DFA(loader)

    pass
