import functools
from collections import defaultdict

import graphviz as gv
from pythomata import SimpleDFA  # help trimming, minimizing & plotting

from States_prefixes import State
from utils import add_nodes, add_edges, timeit
from config import START_PREFIX, SEP
from Fidelity import parse_tree_with_dfa


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

        self.delta = defaultdict(dict)  # transition table

    # todo: use cashed result to accelerate
    def prefix2state(self, prefix):
        """ Return the state in DFA for prefix.

        Note: We should use prefixes to index instead of parsing transition table as when this prefix2state is called,
            the transition table hadn't updated."""
        for state in self.Q:
            if prefix in state.prefixes:
                return state
        raise ValueError("State for prefix %s not found." % prefix)

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
        mapping, missing = parse_tree_with_dfa(rnn_loader.prefix_tree.root, self.q0, self)
        accepted_sup = sum([node.sup for node in mapping[self.F]])
        # assert abs(rnn_loader.prefix_tree.fidelity(accepted_sup) - rnn_loader.eval_fidelity(self)) < 1e-6
        return rnn_loader.prefix_tree.fidelity(accepted_sup)
        # return rnn_loader.eval_fidelity(self)

    def _check_null_states(self):
        reachable_states = {self.q0}
        for state in self.delta.keys():
            reachable_states.update(self.delta[state].values())
        for state in self.Q:
            if state not in reachable_states:
                if state == self.F:
                    raise RuntimeError("Accepting state unreachable.")
                else:
                    self.Q.remove(state)  # Remove unreachable state in dfa
                    for prefix in state.prefixes:
                        self._delete_prefix(state, prefix)
                    if state in self.delta.keys():
                        del self.delta[state]

                    try:
                        if state in self.A_t:
                            self.A_t.remove(state)  # Remove in "states to be merged" queue for consistency
                    except AttributeError:
                        pass

    def _delete_prefix(self, state, prefix):
        """ Delete a prefix of a given state.

        Remove all children from this prefix."""
        if prefix not in state.prefixes:
            raise ValueError("Prefix %s doesn't belong to state." % prefix)
        stack = [(state, prefix)]
        while stack:
            state_, prefix_ = stack.pop()
            state_.prefixes.remove(prefix_)

            if state_ in self.delta.keys():
                for s in self.delta[state_].keys():
                    child = self.delta[state_][s]
                    if prefix_ + [s] in child.prefixes:
                        stack.append((child, prefix_ + [s]))

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
