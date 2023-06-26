from collections import defaultdict


class SymbolNode4Fidelity:
    """ Symbol node with support and (sample) proportion."""

    def __init__(self, val):
        """
        Args:
            val: str, a symbol in the alphabet of the DFA
        """
        self.val = val  # Symbol in the alphabet
        self.next = []
        self.pos_sup, self.neg_sup = 0, 0
        self.pos_prop, self.neg_prop = 0, 0

    @property
    def sup(self):
        """ Support of (the prefix to) the node.

        Math:
            sup = pos_sup - neg_sup
        """
        if self.pos_sup or self.neg_sup:
            return self.pos_sup - self.neg_sup
        raise AttributeError("Node has neither positive support nor negative support.")

    @property
    def prop(self):
        """ Sample proportion of (the prefix to) the node.

        Math:
            prop = pos_prop - neg_prop
        """
        return self.pos_prop - self.neg_prop

    @property
    def is_sample(self):
        """ Sign of whether (the prefix to) the node is a sample."""
        if self.pos_prop != 0 or self.neg_prop != 0:
            return True
        return False


class PrefixTree4Fidelity:
    """ Prefix tree used for pattern sampling & fidelity calculation.

    Each node of the tree is a SymbolNode, with support & proportion calculated.
    """

    def __init__(self, seq, hidden, label):
        """
        Args:
            seq: input_sequence, array of shape (N, PAD_LEN)
            hidden: hidden_states, array of shape (N, PAD_LEN, hidden_dim)
            label: rnn_output, array of shape (N,)
        """
        self.root = SymbolNode4Fidelity('<start>')
        self._pos_weight, self._neg_weight = 1 / len(label), 1 / len(label)
        self._prop = 1 / len(label)

        self._build_tree(seq, hidden, label)

    def _build_tree(self, seq, hidden, label):
        for s, h, l in zip(seq, hidden, label):
            self._update(s, h, l)

    def _update(self, s, h, l):
        cur = self.root
        if l:
            cur.pos_sup += self._pos_weight
            if len(s) == 0:  # an empty sequence
                cur.pos_prop += self._prop
        else:
            cur.neg_sup += self._neg_weight
            if len(s) == 0:  # an empty sequence
                cur.neg_prop += self._prop

        for i, symbol in enumerate(s):
            for n in cur.next:
                if n.val == symbol:
                    cur = n
                    break
            else:
                node = SymbolNode4Fidelity(symbol)
                node.h = h[i, :]
                cur.next.append(node)
                cur = node

            if l:
                cur.pos_sup += self._pos_weight
            else:
                cur.neg_sup += self._neg_weight

        if l:
            cur.pos_prop += self._prop
        else:
            cur.neg_prop += self._prop


def parse_tree_with_dfa(node, state, dfa):
    """ Parse prefix tree with absorb DFA using DFS.

    Args:
        node: SymbolNode4Fidelity, the start node (root of the (sub)tree) to search
        state: State, the state of the DFA corresponding to the node
        dfa: DFA, with absorb=True

    Return:
        state2nodes: dict[list], mapping from states (of the dfa) to list of nodes
        missing: list, list of nodes to which the dfa has no forward transition

    Note:
        In absorb=True cases, we don't have to search deeper when an accept state is reached,
        as all the descendent nodes will belong to the accept state.

        Thus, this one is much faster than the parse_tree_with_non_absorb_dfa function.
    """
    assert dfa.absorb is True, "This parsing is for DFA which absorb=True."

    stack = [(node, state)]
    # state2nodes, missing_nodes = defaultdict(set), set()
    state2nodes = defaultdict(set)
    while stack:
        node_, state_ = stack.pop()
        state2nodes[state_].add(node_)
        if state_ not in dfa.F:  # accept states reached
            for n in node_.next:
                if n.val in dfa.delta[state_].keys():  # transition already in dfa
                    s = dfa.delta[state_][n.val]
                    stack.append((n, s))
                # else:  # either should be negative expression or positive expression misclassified (hasn't added)
                #     missing_nodes.add(n)

    # return state2nodes, missing_nodes
    return state2nodes


def parse_tree_with_non_absorb_dfa(node, state, dfa):
    """ Parse prefix tree with non-absorb DFA using DFS.

    Args:
        node: SymbolNode4Fidelity, the start node (root of the (sub)tree) to search
        state: State, the state of the DFA corresponding to the node
        dfa: DFA, with absorb=False

    Return:
        state2nodes: dict[list], mapping from states (of the dfa) to list of nodes
        missing: list, list of nodes to which the dfa has no forward transition

    Note:
        In absorb=False cases, we have to search deeper when an accept state is reached,
        as the descendent nodes may belong to other states.
    """
    assert dfa.absorb is False, "DFA which absorb=True should use parse_tree_with_dfa."

    stack = [(node, state)]
    # state2nodes, missing_nodes = defaultdict(set), set()
    state2nodes = defaultdict(set)
    while stack:
        node_, state_ = stack.pop()
        state2nodes[state_].add(node_)
        for n in node_.next:
            if n.val in dfa.delta[state_].keys():  # transition already in dfa
                s = dfa.delta[state_][n.val]
                stack.append((n, s))
            # else:  # either should be negative expression or positive expression misclassified (hasn't added)
            #     missing_nodes.add(n)

    # return state2nodes, missing_nodes
    return state2nodes


if __name__ == "__main__":
    from data.utils import load_pickle
    from config import DFA_DIR
    from utils import RNNLoader

    fname, model = 'synthetic_data_1', 'gru'
    loader = RNNLoader(fname, model)
    dfa = load_pickle(DFA_DIR, "{}_{}".format(fname, model))

    mapping, missing = parse_tree_with_dfa(loader.prefix_tree.root, dfa.q0, dfa)

    pass
