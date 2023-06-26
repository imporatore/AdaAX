import copy
import warnings
from collections import defaultdict, deque

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

        absorb: bool, whether the accept states 'absorb' transitions
        A_t: list[State], ready-to-merge states list
        mapping: dict[State: list[Node]], mapping of State to the list of nodes in the PrefixTree
        missing: list[Node], list of missing nodes (transitions) in the DFA

        fidelity: float
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
        self.F = set()  # accept states
        self.Q = [self.q0]  # states

        self.delta = TransitionTable()  # transition table

        self.absorb = absorb  # if accept state absorb transitions

        self.mapping = defaultdict(set)  # state2nodes mapping
        # self.missing = set()  # current missing nodes

        self.fidelity = 0.

    def __copy__(self):
        """ Deep copy for DFA."""
        map_dict = {state: copy.deepcopy(state) for state in self.Q}
        new_dfa = DFA(self.alphabet.copy(), map_dict[self.q0], self.absorb)
        new_dfa.F = {map_dict[state] for state in self.F}
        new_dfa.Q = [map_dict[state] for state in self.Q]
        new_dfa.delta = self.delta.copy_by_mapping(map_dict)

        if hasattr(self, 'A_t'):
            new_dfa.A_t = deque([map_dict[state] for state in self.A_t])
        if hasattr(self, 'mapping'):
            new_dfa.mapping = defaultdict(set, {map_dict[state]: {node for node in self.mapping[state]}
                                                for state in self.mapping.keys()})
        # if hasattr(self, 'missing'):
        #     new_dfa.missing = {node for node in self.missing}
        if hasattr(self, 'fidelity'):
            new_dfa.fidelity = self.fidelity
        return new_dfa

    def __eq__(self, other):
        """ Check if two DFAs are equal (equivalent in accept states & transitions).

        Note:
            q0, Q(except reachable states), A_t, mapping, missing won't be checked.
        """
        if self.alphabet != other.alphabet:  # check alphabet
            return False
        if self.absorb != other.absorb:  # check absorbing behavior
            return False

        # check transition & state mapping: self -> other
        if not self._check_state_transition_mapping(other):
            return False
        # check transition & state mapping: other -> self
        if not other._check_state_transition_mapping(self):
            return False

        return True

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

    def update_node_fidelity(self, rnn_loader, node, class_balanced):
        """ Update fidelity for new node (with its state) accepted."""
        pos_count, neg_count, total_count = rnn_loader.counts

        if self.absorb:
            accepted_pos = node.pos_sup
            accepted_neg = node.neg_sup
        else:
            accepted_pos = node.pos_prop
            accepted_neg = node.neg_prop

        if not class_balanced:
            self.fidelity += accepted_pos - accepted_neg
        else:
            self.fidelity += total_count * accepted_pos / (2 * pos_count) - total_count * accepted_neg / (2 * neg_count)

    def eval_fidelity(self, rnn_loader, class_balanced):
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
        pos_count, neg_count, total_count = rnn_loader.counts

        if self.absorb:
            accepted_pos = sum([sum([node.pos_sup for node in self.mapping[state]]) for state in self.F])
            accepted_neg = sum([sum([node.neg_sup for node in self.mapping[state]]) for state in self.F])
        else:
            accepted_pos = sum([sum([node.pos_prop for node in self.mapping[state]]) for state in self.F])
            accepted_neg = sum([sum([node.neg_prop for node in self.mapping[state]]) for state in self.F])

        if not class_balanced:
            return accepted_pos - accepted_neg + neg_count / total_count
        else:
            return total_count * accepted_pos / (2 * pos_count) - total_count * accepted_neg / (2 * neg_count) + .5

    def parse_tree(self, node, state):
        """ Parse tree with DFA.

        Args:
            node: Node, the node to start parsing (root node of the subtree)
            state: State, the state corresponding to the node
        """
        assert state in self.Q, "State not recognized."
        if self.absorb:
            # mapping, missing = parse_tree_with_dfa(node, state, self)
            mapping = parse_tree_with_dfa(node, state, self)
        else:
            # mapping, missing = parse_tree_with_non_absorb_dfa(node, state, self)
            mapping = parse_tree_with_non_absorb_dfa(node, state, self)
        # return mapping, missing
        return mapping

    # def update_mapping(self, mapping, missing):
    def update_mapping(self, mapping):
        """ Update mapping and missing."""
        for state in mapping.keys():
            self.mapping[state].update(mapping[state])
        # self.missing.update(missing)

    def remove_descendants(self, node, state):
        """ Remove all descendants of a node in dfa state2nodes mapping and missing.

        Notes:
            used when a new state is accepted for absorb=True DFA.
            DO NOT remove the node itself.
        """
        if self.absorb:
            stack = [(node, state)]
            while stack:
                node_, state_ = stack.pop()
                for n in node_.next:
                    if n.val in self.delta[state_].keys():  # transition exist, remove from mapping
                        new_node, new_state = n, self.delta[state_][n.val]
                        # may already been removed from the parsing of predecessor
                        # since the state was not originally an accept state, so there may be ancestors in mapping
                        if new_node in self.mapping[new_state]:
                            self.mapping[new_state].remove(new_node)
                            stack.append((new_node, new_state))
                    # else:
                    #     if n in self.missing:
                    #         self.missing.remove(n)

    def _check_state_transition_mapping(self, other):
        """ Check if there exist a mapping from self to other that all transitions in self exist in other."""
        mapping, stack, visited = {self.q0: other.q0}, [self.q0], set()
        while stack:
            state = stack.pop()
            mapped_state = mapping[state]
            visited.add(state)  # update visited
            for s in self.delta[state].keys():
                if s not in other.delta[mapped_state].keys():  # no transition found
                    return False
                child = self.delta[state][s]
                if child in visited:
                    if mapping[child] != other.delta[mapped_state][s]:  # wrong descendant
                        return False
                else:
                    mapping.update({child: other.delta[mapped_state][s]})  # update mapping for new states
        return True

    def _check_accept_states(self):
        assert all([state in self.Q for state in self.F]), "Accept states not found in states list."

    def _check_absorbing(self):
        """ Check if accept states have exiting transitions."""
        if self.absorb:
            for state in self.F:
                if state in self.delta.keys():
                    warnings.warn("Exiting transitions found in accept state when absorb=True")
                    for node in self.mapping[state].copy():
                        self.remove_descendants(node, state)
                    self.delta.delete_forward(state)

    def _check_null_states(self):
        """ Check if there are unreachable states."""
        reachable_states, stack, visited = {self.q0}, [self.q0], set()

        while stack:
            state = stack.pop()
            visited.add(state)  # update visited
            reachable_states.update(self.delta[state].values())  # update reachable states
            for s in set(self.delta[state].values()):
                if s not in visited and s not in stack:
                    stack.append(s)

        for state in self.Q.copy():
            if state not in reachable_states:
                if state in self.F:
                    warnings.warn("Accept state unreachable.")
                    self.F.remove(state)
                    if not self.F:
                        raise RuntimeError("No accept state remains.")
                for node in self.mapping[state].copy():
                    self.remove_descendants(node, state)
                if state in self.mapping.keys():
                    del self.mapping[state]
                self.delta.pop(state)  # Remove unreachable state in dfa
                self.Q.remove(state)
                if hasattr(self, 'A_t') and state in self.A_t:
                    self.A_t.remove(state)  # Remove state in "states to be merged" queue for consistency

    def _check_transition_consistency(self):
        """ Check if forward and backward transitions are consistent."""
        self.delta._check_transition_consistency()

    def _check_state_consistency(self):
        """ Check if there are unrecognized states in forward transitions."""
        self.delta._check_state_consistency(self.Q)

    def _check_empty_transition(self):
        """ Check if there is empty(null) children state in transitions."""
        self.delta._check_empty_transition()

    def _check_state_node_mapping(self):
        """ Check if the mapping is consistent with states and missing."""
        assert all([state in self.Q for state in self.mapping.keys()]), "State not recognized in state-node mapping."

        # remove empty mapping
        for state in self.mapping.keys():
            if len(self.mapping[state]) == 0:
                del self.mapping[state]

        # for state in self.mapping.keys():
        #     assert all([node not in self.missing for node in self.mapping[state]]), \
        #         "Node in state-node mapping also found in missing."

    # todo: remove the usage of SimpleDFA and implement minimize, complete, trimming
    def to_simpledfa(self, minimize, trim):
        """ Convert into pythomata.SimpleDFA for plot."""
        alphabet = set(self.alphabet)

        if self.q0 in self.F:
            if len(self.F) == 1:
                states_mapping, count = {self.q0: 'Start & Accept'}, 1
            else:
                states_mapping, count, accept_count = {self.q0: 'Start & Accept1'}, 1, 2

                for state in self.F:
                    if state != self.q0:
                        states_mapping.update({state: 'Accept' + str(accept_count)})
                        accept_count += 1
        else:
            states_mapping, count = {self.q0: 'Start'}, 1

            if len(self.F) == 1:
                states_mapping.update({list(self.F)[0]: 'Accept'})
            else:
                states_mapping.update({state: 'Accept' + str(i + 1) for i, state in enumerate(self.F)})

        for state in self.Q:
            if state not in self.F.union({self.q0}):
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
