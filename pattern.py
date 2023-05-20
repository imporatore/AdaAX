import warnings

from sklearn.cluster import KMeans
import numpy as np

from config import K, THETA, START_SYMBOL, START_PREFIX
# todo: check if extracted pattern is unique


# todo: require testing
def pattern_extraction(rnn_loader):
    """ Extract patterns by DFS backtracking at the level of core(focal) sets.

        Args:
            - rnn_loader

        Params:
            - K: Initial cluster numbers, determined by elbow method
            - THETA: threshold for pruning focal set(split by clusters)

        Return:
            - patterns: list[list], list of patterns, each pattern is a list of symbols which
                leads to a core(pure) set, and finally reaches accept state.
            - support: list[float], data support(percentage in the sample) for corresponding pattern
        """

    def _clustering():
        """ Pre-clustering to reduce complexity when backtracking."""
        N, L = rnn_loader.hidden_states.shape[:2]
        # first layer (start state) & last layer (accept state) doesn't participate in the clustering stage
        kmeans = KMeans(n_clusters=K, init='k-means++', n_init='auto').fit(
            rnn_loader.hidden_states[:, len(START_PREFIX): -1, :].reshape((N * (L - 1 - len(START_PREFIX)), -1)))
        # add a cluster -1 for start state
        start_cluster = np.array([-1] * N, dtype=np.int32).reshape((N, 1))
        return np.hstack(start_cluster, kmeans.labels_.reshape((N, (L - 1 - len(START_PREFIX)))))

    clusters = _clustering()  # Notice shape, (N, L-1)
    patterns, support = [], []

    # pattern extraction using index and level to represent hidden states
    def _pattern_extraction(ind, lvl, p):
        """ One step of the pattern extraction procedure,

        use cluster_id and previous symbol to backtrack from a 'focal set',
        which is a 'core set' that shares a common cluster_id and suffix except the first one (accept state).
        Recursively applied to the previous 'focal set' and search for patterns by DFS.

        Args:
            - ind: list, indexes in the input sequence of the hidden states in current 'focal set'
            - lvl: int, current level of the focal set, 0 stands for start state and L-1 stands for accept state
            - p: list, current extracted pattern
        """
        # Reaches start state and add pattern
        # When start symbol was added,
        if lvl == 0:
            patterns.append(START_PREFIX + p)
            support.append(len(ind) / rnn_loader.input_sequences.shape[0])
            return

        # Split previous states by cluster_ids
        # ! Could be moved inside the for loop of cluster to save memory
        inds = {k: [i for i in ind if clusters[i, lvl - 1] == k] for k in range(-1, K)}
        # Sort cluster ids by the size each sub cluster
        cluster_ids = sorted(range(-1, K), key=lambda x: len(inds[x]), reverse=True)

        for k in cluster_ids:

            # Prune if the size of sub cluster is too small
            # ? Actually we are not calculating the data support of core set, instead we sum by symbols
            # ! Use break instead of continue since we have already sorted cluster_ids by its size in descent order
            if len(inds[k]) / rnn_loader.input_sequences.shape[0] < THETA:
                break

            # ! Likewise, move it outside the for loop for accelerating
            Hs = {}
            for i in inds[k]:
                # Symbols backtracked from hidden_states
                # ! Moved it inside the loop since the symbols used by sub clusters
                # may not correspond to the whole prev states
                Hs[rnn_loader.input_sequences[i, lvl]] = Hs.get(rnn_loader.input_sequences[i, lvl]) + [i]

            # Search each symbol in descent order
            # ? Prune trivial symbols?
            for x in sorted(Hs.keys(), key=lambda x: len(Hs[x]), reverse=True):
                # prepend symbol to the current pattern
                _pattern_extraction(Hs[x], lvl - 1, [x] + p)

    # Start fom the first focal set (last hidden layer of positive input sequences)
    pos_ind = np.arange(rnn_loader.input_sequences.shape[0])[rnn_loader.rnn_output == 1]
    _pattern_extraction(pos_ind, rnn_loader.input_sequences.shape[1] - 1, [])
    return patterns, support


class SymbolNode:

    def __init__(self, val):
        self.val = val  # Symbol in the alphabet
        self.next = []


# todo: require testing
# Yet, the PatternTree is analogous to building a DFA without the consolidation phase for now.
# Unused for now, can be used to visualize patterns and corresponding support.
class PatternTree:
    """ Prefix tree for extracted pattern."""

    def __init__(self, patterns, support):
        self.root = SymbolNode(START_SYMBOL)
        self._build_tree(patterns, support)

    def _build_tree(self, patterns, support):
        for p, s in zip(patterns, support):
            self._update(p[len(START_PREFIX):], s)  # if START_SYMBOL, the first symbol of pattern would be START_SYMBOL


    # Seems there can't be two same pattern extracted.
    # todo: modify the code.
    def _update(self, p, s):
        cur = self.root
        for symbol in p:
            for n in cur.next:
                if n.val == symbol:
                    cur = n
                    break
            else:
                node = SymbolNode(symbol)
                cur.next.append(node)
                cur = node
        try:
            warnings.warn('Find existing leaf node (pattern), support summed.')
            cur.sup += s  # add support to existing leaf node
        except AttributeError:
            cur.sup = s  # assign support to new leaf node

    # todo: PatternTree flow
    # so that we don't have to start from the start state when adding new pattern
    def __iter__(self):
        yield


if __name__ == "__main__":
    # todo: func test: pattern_extraction
    # todo: class test: PatternTree
    pass
