from collections import defaultdict

from config import START_SYMBOL, START_PREFIX


class SymbolNode4Fidelity(object):

    def __init__(self, val):
        self.val = val  # Symbol in the alphabet
        self.next = []
        self.pos_sup = 0
        self.neg_sup = 0
        self.pos_prop = 0
        self.neg_prop = 0

    @property
    def sup(self):
        if self.pos_sup or self.neg_sup:
            return self.pos_sup - self.neg_sup
        raise AttributeError("Node has neither positive support nor negative support.")

    @property
    def prop(self):
        return self.pos_prop - self.neg_prop

    @property
    def is_sample(self):
        if self.pos_prop != 0 or self.neg_prop != 0:
            return True
        return False


class PrefixTree4Fidelity:

    def __init__(self, seq, hidden, label):
        self.root = SymbolNode4Fidelity(START_SYMBOL)
        self.root.h = hidden[0, 0, :]  # hidden values for start symbol
        self._pos_weight, self._neg_weight = 1 / len(label), 1 / len(label)
        self._prop = 1 / len(label)

        self._build_tree(seq, hidden, label)

    def _build_tree(self, seq, hidden, label):
        for s, h, l in zip(seq, hidden, label):
            self._update(s[len(START_PREFIX):], h[len(START_PREFIX):], l)

    def _update(self, s, h, l):
        cur = self.root
        if l:
            cur.pos_sup += self._pos_weight
            if len(s) == 0:
                cur.pos_prop += self._prop
        else:
            cur.neg_sup += self._neg_weight
            if len(s) == 0:
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

            if i == len(s) - 1 or s[i + 1] == '<PAD>':  # reached the end of a sample
                if l:
                    cur.pos_prop += self._prop
                else:
                    cur.neg_prop += self._prop

    def eval_hidden(self, expr):
        cur = self.root
        for symbol in expr:
            for n in cur.next:
                if n.val == symbol:
                    cur = n
        return cur.h

    def __iter__(self):
        """ Parse the tree using DFS.

        Only yield positive samples in rnn result.

        *** Bad performance ***

        Note:
            No pattern is another pattern's prefix.

        Return:
            expr: list, list of symbols, positive patterns
            h: list, list of hidden values in expr
            sup: tuple(float, float), positive and negative support of the pattern
        """
        stack = [(self.root, [self.root.val], [self.root.h])]
        while stack:
            node, expr, hidden = stack.pop()
            try:
                if node.pos_prop:
                    yield expr, hidden, (node.pos_sup, node.neg_sup)
            except AttributeError:
                for n in node.next:
                    if n.val != '<PAD>':
                        stack.append((n, expr + [n.val], hidden + [n.h]))


def parse_tree_with_dfa(node, state, dfa):
    """

    Args:
        node:
        state:
        dfa:

    Return:

    """
    assert dfa.absorb is True, "This fidelity parsing is for absorb DFA."

    stack = [(node, state)]
    state2nodes, missing_nodes = defaultdict(list), []
    while stack:
        node_, state_ = stack.pop()
        state2nodes[state_].append(node_)
        # if (node_ == dfa.q0 and dfa.q0 == dfa.F) or state_ != dfa.F:  # accept state reached
        if state_ != dfa.F:  # accept state reached
            for n in node_.next:  # transition already in dfa
                if n.val in dfa.delta[state_].keys():
                    s = dfa.delta[state_][n.val]
                    stack.append((n, s))
                elif n.val == '<pad>':  # reach the end of an expression (symbols all found but classified as negative)
                    pass
                else:  # either should be negative expression or positive expression misclassified (hasn't added)
                    missing_nodes.append(n)

    return state2nodes, missing_nodes


def parse_tree_with_non_absorb_dfa(node, state, dfa):
    """

    Note:
        much slower than absorbed dfa.

    Args:
        node:
        state:
        dfa:

    Return:

    """
    assert dfa.absorb is False, "This fidelity parsing is for non-absorb DFA."

    stack = [(node, state)]
    state2nodes, missing_nodes = defaultdict(list), []
    while stack:
        node_, state_ = stack.pop()
        state2nodes[state_].append(node_)
        for n in node_.next:  # transition already in dfa
            if n.val in dfa.delta[state_].keys():
                s = dfa.delta[state_][n.val]
                stack.append((n, s))
            elif n.val == '<pad>':  # reach the end of an expression (symbols all found but classified as negative)
                pass
            else:  # either should be negative expression or positive expression misclassified (hasn't added)
                missing_nodes.append(n)

    return state2nodes, missing_nodes


if __name__ == "__main__":
    from data.utils import load_pickle
    from config import DFA_DIR
    from utils import RNNLoader

    fname, model = 'synthetic_data_1', 'gru'
    loader = RNNLoader(fname, model)
    dfa = load_pickle(DFA_DIR, "{}_{}".format(fname, model))

    mapping, missing = parse_tree_with_dfa(loader.prefix_tree.root, dfa.q0, dfa)

    pass
