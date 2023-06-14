import warnings

from pythomata import SimpleDFA  # help trimming, minimizing & plotting

from States import State
from Transition import TransitionTable
from Fidelity import parse_tree_with_dfa, parse_tree_with_non_absorb_dfa


# todo: add state split to ensure gradually learning more & more difficult patterns from flow or samplers
class DFA:
    """ DFA for AdaAX.

    Attributes:
        alphabet: list, alphabet (set of symbols) used in the DFA
        q0: State, start state of the DFA
        F: list[State], list of accept states of the DFA
        Q: list[State], list of states of the DFA
        delta: TransitionTable, transition table of the DFA
    """

    def __init__(self, alphabet, start_state, absorb):
        """
        Args:
            alphabet: list, alphabet (set of symbols) used in the DFA
            start_state: State, start state of the DFA
            absorb: bool, whether the accept states 'absorb' transitions
        """
        self.alphabet = alphabet  # alphabet
        self.q0 = start_state  # start state
        self.F = []  # accept states
        self.Q = [self.q0]  # states

        self.delta = TransitionTable()  # transition table

        self.absorb = absorb  # if accept state absorb transitions

    def add_new_state(self, prefix, hidden, prev=None):
        """ Add and return the new state from a new prefix.

        Args:
            prefix: list, a list of symbols which initialize the State (as a PureSet).
            hidden: float, hidden values of 'prefix'
            prev: None or State, parent state of the nre state
        """
        state = State(hidden_values=hidden)  # Initialize pure set from new prefix
        self.Q.append(state)  # Add to states

        if prev:
            self.add_transit(prev, prefix[-1], state)  # Update transition

        return state

    def add_transit(self, state1, symbol, state2):
        """ Add a transition from state1 to state2 by symbol."""
        self.delta[state1][symbol] = state2  # update transition table

    def classify_expression(self, expression):
        """ Classify expression using the DFA.

        Args:
            expression: list, list of symbols with only those in the alphabet

        Note:
            For absorb=True DFAs, once a transition goes to accept states, the expression is classified 'positive'.
            For absorb=False DFAs, the expression is classified 'positive' iff the state
                of the last symbol in the expression are among the accept states.
            Missing transitions goes to 'sink' state and classified as 'negative'.
        """
        assert all([s in self.alphabet for s in expression])

        q = self.q0
        for s in expression:
            if s in self.delta[q].keys():
                q = self.delta[q][s]
            else:  # if no transition found, then expression goes to 'sink' state and classified as 'negative'
                return False
            if self.absorb and q in self.F:
                return True
        if self.absorb:
            return False
        else:
            return q in self.F

    def fidelity(self, rnn_loader, class_balanced=False):
        """
        Args:
            rnn_loader: RNNLoader
            class_balanced: bool, default=False, whether to calculate fidelity using class balanced weights

        Math:
            Note that all expressions which is not accepted is classified as negative,
            so we may calculate fidelity using these four parts:

            for absorb=True:
                fidelity = accepted_pos_sup + unaccepted_neg_sup
            for absorb=False:
                fidelity = accepted_pos_prop + unaccepted_neg_prop

            when class_balanced=True:
                pos_sup (and pos_prop) are re-weighted to (total_samples) / (2 * positive_samples)
                neg_sup (and neg_prop) are re-weighted to (total_samples) / (2 * negative_samples)

        Note:
            set class_balanced to true may cause great misbehavior of AdaAX.

            For Tomita 4, linking start and accept states with '1' would cause magnificent fidelity loss,
                as most samples, including those which started with '1' are negative samples.

                However, when class_balanced=True, those positive samples are over-weighted to
                (total_samples) / (2 * positive_samples), thus may not result in fidelity loss for this misbehavior.
        """

        pos_count, total_count = sum(rnn_loader.rnn_output), len(rnn_loader.rnn_output)
        neg_count = total_count - pos_count

        if self.absorb:
            mapping, missing = parse_tree_with_dfa(rnn_loader.prefix_tree.root, self.q0, self)
            accepted_pos = sum([sum([node.pos_sup for node in mapping[state]]) for state in self.F])
            accepted_neg = sum([sum([node.neg_sup for node in mapping[state]]) for state in self.F])
        else:
            mapping, missing = parse_tree_with_non_absorb_dfa(rnn_loader.prefix_tree.root, self.q0, self)
            accepted_pos = sum([sum([node.pos_prop for node in mapping[state]]) for state in self.F])
            accepted_neg = sum([sum([node.neg_prop for node in mapping[state]]) for state in self.F])

        if not class_balanced:
            return accepted_pos - accepted_neg + neg_count / total_count
        else:
            return total_count * accepted_pos / (2 * pos_count) - total_count * accepted_neg / (2 * neg_count) + .5

    def _check_absorbing(self):
        """ Check if accept states have exiting transitions."""
        if self.absorb:
            for state in self.F:
                if state in self.delta.keys():
                    warnings.warn("Exiting transitions found in accept state when absorb=True")
                    self.delta.delete_forward(state)

    def _check_null_states(self):
        """ Check if there are unreachable states."""
        reachable_states, stack, visited = {self.q0}, [self.q0], []

        while stack:
            state = stack.pop()
            visited.append(state)  # update visited
            reachable_states.update(self.delta[state].values())  # update reachable states
            for s in set(self.delta[state].values()):
                if s not in visited and s not in stack:
                    stack.append(s)

        for state in self.Q.copy():
            if state not in reachable_states:
                if state in self.F:
                    warnings.warn("Accept state unreachable.")
                    self.F.remove(state)
                self.delta.pop(state)  # Remove unreachable state in dfa
                self.Q.remove(state)
                try:
                    if state in self.A_t:
                        self.A_t.remove(state)  # Remove state in "states to be merged" queue for consistency
                except AttributeError:
                    pass

    def _check_transition_consistency(self):
        """ Check if forward and backward transitions are consistent."""
        self.delta._check_transition_consistency()

    def _check_state_consistency(self):
        """ Check if there are unrecognized states in forward transitions."""
        self.delta._check_state_consistency(self.Q)

    def _check_empty_transition(self):
        """ Check if there is empty(null) children state in transitions."""
        self.delta._check_empty_transition()

    # todo: remove the usage of SimpleDFA and implement minimize, complete, trimming
    def to_simpledfa(self, minimize, trim):
        """ Convert into pythomata.SimpleDFA for plot."""
        alphabet = set(self.alphabet)

        # todo: q0 in F (start state among accept states)
        states_mapping, count = {self.q0: 'Start'}, 1
        states_mapping.update({state: 'Accept' + str(i + 1) for i, state in enumerate(self.F)})

        for state in self.Q:
            if state not in [self.q0] + self.F:
                states_mapping.update({state: 'State' + str(count)})
                count += 1

        states = set([states_mapping[state] for state in self.Q])
        initial_state = states_mapping[self.q0]
        accept_states = set(states_mapping[state] for state in self.F)
        transition_function = {states_mapping[state]: {symbol: states_mapping[
            s] for symbol, s in self.delta[state].items()} for state in self.delta.keys()}
        dfa = SimpleDFA(states, alphabet, initial_state, accept_states, transition_function)

        if minimize:
            dfa = dfa.minimize()
        if trim:
            dfa = dfa.trim()

        return dfa

    def plot(self, fname, minimize=False, trim=False):
        graph = self.to_simpledfa(minimize=minimize, trim=trim).to_graphviz()
        graph.render(filename=fname, format='png')


if __name__ == "__main__":
    pass
