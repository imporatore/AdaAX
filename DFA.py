import functools

import graphviz as gv
from pythomata import SimpleDFA  # help trimming, minimizing & plotting

from States import State
from Transition import TransitionTable
from utils import add_nodes, add_edges, timeit
from config import START_PREFIX, SEP

# digraph = functools.partial(gv.Digraph, format='png')
# graph = functools.partial(gv.Graph, format='png')


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

        self.delta = TransitionTable()  # transition table

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

    def classify_expression(self, expression):
        """

        Note: Missing transitions goes to 'sink' state and classified as Negative.
        """
        # Expression is string with only letters in alphabet
        q = self.q0
        for s in expression[len(START_PREFIX):]:
            if s in self.delta[q].keys():
                q = self.delta[q][s]
            else:  # if no transition found, then expression is not among the extracted patterns
                return False
            if q == self.F:
                return True
        return False

    @timeit
    def fidelity(self, rnn_loader):
        return rnn_loader.eval_fidelity(self)

    # todo: remove the usage of SimpleDFA and implement minimize, complete, trimming
    def to_simpledfa(self, minimize, trim):
        alphabet = set(self.alphabet)
        states_mapping = {state: 'state' + str(i + 1) for i, state in enumerate(self.Q)}
        states = set([states_mapping[state] for state in self.Q])
        initial_state = states_mapping[self.q0]
        accepting_states = {states_mapping[self.F]}

        transition_function = {states_mapping[state]: {symbol: states_mapping[
            s] for symbol, s in self.delta[state].items()} for state in self.delta.keys()}  # if state != self.F
        dfa = SimpleDFA(states, alphabet, initial_state, accepting_states, transition_function)

        # if minimize:
        #     dfa = dfa.minimize()
        # if trim:
        #     dfa = dfa.trim()

        return dfa

    def plot(self, fname, minimize=True, trim=True):
        graph = self.to_simpledfa(minimize=minimize, trim=trim).to_graphviz()
        graph.render(filename=fname, format='png')

    # todo: require testing
    # def plot(self, fname, force=False, maximum=60):
    #
    #     edges = defaultdict(list)
    #
    #     for parent in self.delta.keys():
    #         for s in self.delta[parent].keys():
    #             child = self.delta[parent][s]
    #             edges[(parent, child)] += [s]
    #
    #     if (not force) and len(self.Q) > maximum:
    #         raise Warning('State number exceeds limit (Maximum %d).' % maximum)
    #
    #     state2int = {None: 0}
    #
    #     def _state2int(state):
    #         if state not in state2int.keys():
    #             state2int[state] = max(state2int.values()) + 1
    #         return str(state2int[state])
    #
    #     g = digraph()
    #     g = add_nodes(g, [(_state2int(self.q0), {'color': 'black', 'shape': 'hexagon', 'label': 'Start'})])
    #     g = add_nodes(g, [(_state2int(state), {'color': 'black', 'label': str(_state2int(state))})
    #                       for state in self.Q if state not in (self.q0, self.F)])
    #     g = add_nodes(g, [(_state2int(self.F), {'color': 'green', 'shape': 'hexagon', 'label': 'Accept'})])
    #
    #     g = add_edges(g, [(e, {'label': SEP.join(edges[e])}) for e in edges.keys()])
    #
    #     display(Image(filename=g.render(filename=fname)))


if __name__ == "__main__":
    pass
