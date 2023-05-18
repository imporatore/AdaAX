from sklearn.cluster import KMeans

from config import K, THETA


def pattern_extraction(input_sequences, hidden_states):
    def _clustering():
        N, L = hidden_states.shape[:2]
        kmeans = KMeans(n_clusters=K, init='k-means++', n_init='auto').fit(hidden_states.reshape((N * L, -1)))
        return kmeans.labels_.reshape((N, L))

    clusters = _clustering()
    patterns, support = [], []

    def _pattern_extraction(ind, lvl, p):
        if lvl == 0:
            patterns.append(p)
            support.append(len(ind) / input_sequences.shape[0])
            return

        inds = [[i for i in ind if clusters[i, lvl - 1] == k] for k in range(K)]
        cluster_ids = sorted(range(K), key=lambda x: len(inds[x]), reverse=True)

        for k in cluster_ids:

            if len(inds[k]) / input_sequences.shape[0] < THETA:
                break

            Hs = {}
            for i in inds[k]:
                Hs[input_sequences[i, lvl - 1]] = Hs.get(input_sequences[i, lvl - 1]) + [i]

            for x in sorted(Hs.keys(), key=lambda x: len(Hs[x]), reverse=True):
                _pattern_extraction(Hs[x], lvl - 1, [x] + p)

    _pattern_extraction(range(input_sequences.shape[0]), input_sequences.shape[1] - 1, [])
    return patterns, support


class SymbolNode:

    def __init__(self, val):
        self.val = val  # Symbol in the alphabet
        self.next = []


class PatternTree:

    def __init__(self, patterns, support):
        self.root = SymbolNode('<START>')
        self._build_tree(patterns, support)

    def _build_tree(self, patterns, support):
        for p, s in zip(patterns, support):
            self._update(p, s)

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
            cur.sup += s  # add support to existing leaf node
        except AttributeError:
            cur.sup = s  # assign support to new leaf node


if __name__ == "__main__":
    pass
