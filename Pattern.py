import warnings

import numpy as np
from sklearn.cluster import KMeans

from config import K, THETA, START_PREFIX, SEP


def pattern_extraction(rnn_loader, remove_padding=True, label=True):
    """ Extract patterns by DFS backtracking at the level of core(focal) sets.

        Args:
            rnn_loader
            remove_padding

        Params:
            K: Initial cluster numbers, determined by elbow method
            THETA: threshold for pruning focal set(split by clusters)

        Return:
            patterns: list[list], list of patterns, each pattern is a list of symbols which
                leads to a core(pure) set, and finally reaches accept state.
            support: list[float], data support(percentage in the sample) for corresponding pattern
        """

    def _clustering():
        """ Pre-clustering to reduce complexity when backtracking."""
        N, L = rnn_loader.hidden_states.shape[:2]
        # first layer (start state) & last layer (accept state) doesn't participate in the clustering stage
        kmeans = KMeans(n_clusters=K, init='k-means++', n_init='auto').fit(
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
        inds = {k: [i for i in ind if clusters[i, lvl - 1] == k] for k in range(-1, K)}
        # Sort cluster ids by the size each sub cluster
        cluster_ids = sorted(inds.keys(), key=lambda x: len(inds[x]), reverse=True)

        for k in cluster_ids:

            # Prune if the size of sub cluster is too small
            # ? Actually we are not calculating the data support of core set, instead we sum by symbols
            # ! Use break instead of continue since we have already sorted cluster_ids by its size in descent order
            if len(inds[k]) / rnn_loader.decoded_input_seq.shape[0] < THETA:
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


# Yet, the PatternTree is analogous to building a DFA without the consolidation phase for now.
# Unused for now, can be used to visualize patterns and corresponding support.
class PatternTree:
    """ Prefix tree for extracted pattern, with support updated."""

    def __init__(self, prefix_tree):
        self.tree = prefix_tree
        self.root = prefix_tree.root

    def update_patterns(self, patterns, support, label=True):
        for p, s in zip(patterns, support):
            # if START_SYMBOL, the first symbol of pattern would be START_SYMBOL
            self._update(p[len(START_PREFIX):], s, label=label)

    # Seems there can't be two same pattern extracted.
    # todo: modify the code.
    def _update(self, p, s, label):
        cur = self.root
        for symbol in p:
            for n in cur.next:
                if n.val == symbol:
                    cur = n
                    if label:
                        try:
                            cur.pos_sup += s
                        except AttributeError:
                            cur.pos_sup = s
                    else:
                        try:
                            cur.neg_sup += s
                        except AttributeError:
                            cur.neg_sup = s
                    break
            else:
                warnings.warn("Node for pattern %s not found." % p)

    def eval_hidden(self, s):
        return self.tree.eval_hidden(s)

    # so that we don't have to start from the start state when adding new pattern
    def __iter__(self):
        """ Parse the tree using DFS.

        Return:

        """
        stack = [(self.root, self.root.val)]
        while stack:
            node, expr = stack.pop()
            if not node.next:
                yield expr, node.pos_sup, node.neg_sup
            else:
                for n in node.next:
                    if n.val == '<pad>':
                        yield expr, node.pos_sup, node.neg_sup
                    else:
                        stack.append((n, expr + SEP + n.val))


if __name__ == "__main__":
    from utils import RNNLoader

    loader = RNNLoader('tomita_data_1', 'gru')
    pos_patterns, pos_supports = pattern_extraction(loader, label=True)
    neg_patterns, neg_supports = pattern_extraction(loader, label=False)
    pattern_tree = PatternTree(loader.prefix_tree)
    pattern_tree.update_patterns(pos_patterns, pos_supports, label=True)
    pattern_tree.update_patterns(neg_patterns, neg_supports, label=False)
    for res in pattern_tree:
        print(res)
    pass
