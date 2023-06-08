import warnings

import numpy as np
from sklearn.cluster import KMeans

from config import START_PREFIX


def pattern_extraction(rnn_loader, cluster_num, pruning, remove_padding=True, label=True):
    """ Extract patterns by DFS backtracking at the level of core(focal) sets.

        Args:
            rnn_loader: RNN data loader.
            remove_padding: bool, if True, then remove the padding symbol in extracted patterns.
            label: bool, True for adding positive patterns and vice versa.

        Params:
            cluster_num: Initial cluster numbers, determined by elbow method
            pruning: threshold for pruning focal set(split by clusters)

        Return:
            patterns: list[list], list of patterns, each pattern is a list of symbols which
                leads to a core(pure) set, and finally reaches accept state.
            support: list[float], data support(percentage in the sample) for corresponding pattern

        Note:
            Extracted patterns are unique.
        """

    def _clustering():
        """ Pre-clustering to reduce complexity when backtracking."""
        N, L = rnn_loader.hidden_states.shape[:2]
        # first layer (start state) & last layer (accept state) doesn't participate in the clustering stage
        kmeans = KMeans(n_clusters=cluster_num, init='k-means++', n_init='auto').fit(
            rnn_loader.hidden_states[:, len(START_PREFIX): -1, :].reshape((N * (L - 1 - len(START_PREFIX)), -1)))
        # add a cluster -1 for start state
        start_cluster = np.array([-1] * N, dtype=np.int32).reshape((N, 1))
        return np.hstack((start_cluster, kmeans.labels_.reshape((N, (L - 1 - len(START_PREFIX))))))

    clusters = _clustering()  # Notice shape, (N, L-1)
    patterns, support = [], []

    # pattern extraction using index and level to represent hidden states
    def _pattern_extraction(ind, lvl, p):
        """ One step of the pattern extraction procedure,

        use cluster_id and previous symbol to backtrack from a 'focal set',
        which is a 'core set' that shares a common cluster_id and suffix except the first one (accept state).
        Recursively applied to the previous 'focal set' and search for patterns by DFS.

        Args:
            ind: list, indexes in the input sequence of the hidden states in current 'focal set'
            lvl: int, current level of the focal set, 0 stands for start state and L-1 stands for accept state
            p: list, current extracted pattern
        """
        # Reaches start state and add pattern
        # When start symbol was added,
        if lvl == 0:
            if START_PREFIX:
                patterns.append(START_PREFIX + p)
            else:
                patterns.append(p)
            support.append(len(ind) / rnn_loader.decoded_input_seq.shape[0])
            return

        # Split previous states by cluster_ids
        # ! Could be moved inside the for loop of cluster to save memory
        inds = {k: [i for i in ind if clusters[i, lvl - 1] == k] for k in range(-1, cluster_num)}
        # Sort cluster ids by the size each sub cluster
        cluster_ids = sorted(inds.keys(), key=lambda x: len(inds[x]), reverse=True)

        for k in cluster_ids:

            # Prune if the size of sub cluster is too small
            # ? Actually we are not calculating the data support of core set, instead we sum by symbols
            # ! Use break instead of continue since we have already sorted cluster_ids by its size in descent order
            if len(inds[k]) / rnn_loader.decoded_input_seq.shape[0] < pruning:
                break

            # ! Likewise, move it outside the for loop for accelerating
            Hs = {}
            for i in inds[k]:
                # Symbols backtracked from hidden_states
                # ! Moved it inside the loop since the symbols used by sub clusters
                # may not correspond to the whole prev states
                Hs[rnn_loader.decoded_input_seq[i, lvl]] = Hs.get(rnn_loader.decoded_input_seq[i, lvl], []) + [i]

            # Search each symbol in descent order
            # ? Prune trivial symbols?
            for x in sorted(Hs.keys(), key=lambda x: len(Hs[x]), reverse=True):
                # prepend symbol to the current pattern
                _pattern_extraction(Hs[x], lvl - 1, [x] + p)

    # Start fom the first focal set (last hidden layer of positive input sequences)
    if label:
        pos_ind = np.arange(rnn_loader.decoded_input_seq.shape[0])[rnn_loader.rnn_output == 1]
        _pattern_extraction(pos_ind, rnn_loader.decoded_input_seq.shape[1] - 1, [])
    else:
        neg_ind = np.arange(rnn_loader.decoded_input_seq.shape[0])[rnn_loader.rnn_output == 0]
        _pattern_extraction(neg_ind, rnn_loader.decoded_input_seq.shape[1] - 1, [])

    if remove_padding:
        for i in range(len(patterns)):
            try:
                patterns[i] = patterns[i][:patterns[i].index('<pad>')]
            except ValueError:
                pass

    # patterns = [rnn_loader.decode(pattern, as_list=True) for pattern in patterns]

    return patterns, support


class PatternTree:
    """ Prefix tree for extracted pattern, with both positive & negative support updated."""

    def __init__(self, prefix_tree):
        self.root = prefix_tree.root

    def update_patterns(self, patterns, support, label=True):
        for p, s in zip(patterns, support):
            # if START_SYMBOL, the first symbol of pattern would be START_SYMBOL
            self._update(p[len(START_PREFIX):], s, label=label)

    def _update(self, p, s, label):
        cur = self.root
        for symbol in p:
            for n in cur.next:
                if n.val == symbol:
                    cur = n
                    if label:
                        cur.pos_sup += s
                    else:
                        cur.neg_sup += s
                    break
            else:
                warnings.warn("Node for pattern %s not found." % p)

    # todo: No pattern is another pattern's prefix.
    def __iter__(self):
        """ Parse the tree using DFS.

        Note:
            No pattern is another pattern's prefix.

        Return:
            expr: list, list of symbols
            h: list, list of hidden values in expr
            sup: tuple(float, float), positive and negative support of expr
        """
        stack = [(self.root, [self.root.val], [self.root.h])]
        while stack:
            node, expr, hidden = stack.pop()
            if not node.next:
                yield expr, hidden, node.pos_sup, node.neg_sup
            else:
                for n in node.next:
                    if n.val == '<pad>':
                        yield expr, hidden, node.pos_sup, node.neg_sup
                    else:
                        stack.append((n, expr + [n.val], hidden + [n.h]))


# todo
class PositivePatternTree(PatternTree):

    def _update(self, p, s, label=True):
        cur = self.root
        for symbol in p:
            for n in cur.next:
                if n.val == symbol:
                    cur = n
                    cur.pos_sup += s
                    break
            else:
                warnings.warn("Node for pattern %s not found." % p)
        cur._pos_pat = True

    def __iter__(self):
        """ Parse the tree using DFS.

        Note:
            No pattern is another pattern's prefix.

        Return:
            expr: list, list of symbols, positive patterns
            h: list, list of hidden values in expr
            sup: float, positive and negative support of the pattern
        """
        stack = [(self.root, [self.root.val], [self.root.h])]
        while stack:
            node, expr, hidden = stack.pop()
            try:
                if node._pos_pat:
                    yield expr, hidden, node.pos_sup
            except AttributeError:
                for n in node.next:
                    stack.append((n, expr + [n.val], hidden + [n.h]))


class PatternIterator:

    def __init__(self, prefix_tree, patterns, support=None):
        self.root = prefix_tree.root
        self.patterns = patterns
        if support:
            self.support = support
        else:
            self.support = [0.] * len(support)

    def __iter__(self):
        for
        cur, hidden = self.root
        for symbol in p:
            for n in cur.next:
                if n.val == symbol:
                    cur = n
                    cur.pos_sup += s
                    break
            else:
                warnings.warn("Node for pattern %s not found." % p)
        cur._pos_pat = True


if __name__ == "__main__":
    from utils import RNNLoader

    loader = RNNLoader('tomita_data_1', 'gru')
    pos_patterns, pos_supports = pattern_extraction(loader, label=True)
    # neg_patterns, neg_supports = pattern_extraction(loader, label=False)

    # pattern_tree = PatternTree(loader.prefix_tree)
    # pattern_tree.update_patterns(pos_patterns, pos_supports, label=True)
    # pattern_tree.update_patterns(neg_patterns, neg_supports, label=False)

    # for res in pattern_tree:
    #     print(res)

    pos_pattern_tree = PositivePatternTree(loader.prefix_tree)
    pos_pattern_tree.update_patterns(pos_patterns, pos_supports)
    for res in pos_pattern_tree:
        print(res)
    pass
