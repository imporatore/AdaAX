from collections import deque


class PatternSampler:

    def __init__(self, rnn_loader, absorb, pos_threshold, sample_threshold, return_sample):
        """
        Args:
            rnn_loader: RNNLoader
            absorb: bool, if this pattern is to build an absorb DFA,
                and thence the pattern will (not) have an absorbing behavior.
            pos_threshold: float, default 0.95, threshold for positive patterns, default 0.95,
                i.e. pos_sup / (pos_sup + neg_sup) > pos_threshold
                - should be adjusted for unbalanced dataset
            sample_threshold: int, default 5, threshold for a positive pattern,
                the minimum number of positive samples from this pattern,
                i.e. pos_sup >= sample_threshold / total_samples
            return_sample: bool, default False, if sampler yield single positive sample
                - if True ana absorb=False, sampler will yield all positive samples in the dataset

        Note:
            two cases for a pattern yielded by pattern sampler
            1. (if return_sample=True) all positive samples
            2. positive patterns, iff
               - pos_sup / (pos_sup + neg_sup) > pos_threshold
               - pos_sup >= sample_threshold / total_samples

            'absorb' will cause different behavior of patterns:
            - for absorb=True DFA, whenever a positive pattern is reached,
                the expression will be classified as positive,
                thus no pattern will be other patterns' prefix
            - for absorb=False DFA, an expression could be false even if it has a positive pattern as prefix.
                Thus pattern sampler won't stop searching when positive patterns were met.
        """
        self.root = rnn_loader.prefix_tree.root
        self._absorb = absorb
        self._threshold, self._sample_threshold = pos_threshold, sample_threshold
        self._total_samples = rnn_loader.rnn_output.shape[0]
        self._return_sample = return_sample

    def __iter__(self):
        """ Parse the tree using BFS (so that simpler/shorter patterns come first)

        Idea:
            use numeric values for sorting: score = f(length, pos_prop)
                - score \propto 1 / length
                - score \propto pos_prop
                i.e. score := length * log(pos_prop)
                    - require Laplace smoothing

        Note:
            this threshold method doesn't work well for negative pattern, e.g. Tomita 4: '000' not in expression
                - as even '1111' can be followed by '000', and thus wasn't considered a positive pattern
                - may be resolved by using '<pad>' to locate (seems to suit both absorb=True & absorb=False)
                - this serve as a very good 'twisting' test
            No pattern is another patterns' prefix (?)

        Return:
            nodes: list[Node], list of nodes of the positive pattern
            expr: list, list of symbols, positive patterns
            hidden: list, list of hidden values in expr
            sup: tuple(float, float), positive and negative support of the pattern
        """
        queue = deque([(self.root, [], [], [])])
        while queue:
            node, nodes, expr, hidden = queue.popleft()

            if self._absorb:
                if self._is_pos(node):
                    yield nodes, expr, hidden, (node.pos_sup, node.neg_sup)
                else:
                    for n in node.next:
                        queue.append((n, nodes + [n], expr + [n.val], hidden + [n.h]))
            else:
                if self._is_pos(node):
                    yield nodes, expr, hidden, (node.pos_sup, node.neg_sup)
                for n in node.next:
                    queue.append((n, nodes + [n], expr + [n.val], hidden + [n.h]))

    def _is_positive_pattern(self, node):
        if node.pos_sup / (node.pos_sup + node.neg_sup) >= self._threshold and node.pos_sup > \
                self._sample_threshold / self._total_samples:
            return True
        return False

    @staticmethod
    def _is_positive_sample(node):
        if node.pos_prop > 0:
            return True
        else:
            return False

    def _is_pos(self, node):
        if not self._return_sample:
            return self._is_positive_pattern(node)
        return self._is_positive_pattern(node) or self._is_positive_sample(node)


class PatternInputer:
    """ Simple iterator for testing external patterns."""

    def __init__(self, rnn_loader, patterns, support=None):
        """
        Args:
            rnn_loader: RNNLoader
            patterns: list[list], list of external patterns
            support: list, list of pattern supports
        """
        self.root = rnn_loader.prefix_tree.root
        self.patterns = patterns
        if support:
            self.support = support
        else:
            self.support = [0.] * len(patterns)

    def __iter__(self):
        """
        Return:
            nodes: list[Node], list of nodes of the positive pattern
            pattern: list, list of symbols, external patterns which can be found in the rnn_loader
            hidden: list, list of hidden values of the pattern
            support: tuple(float, float), positive and negative support of the pattern
        """
        for pattern, support in zip(self.patterns, self.support):
            cur, nodes, hidden = self.root, [], []
            for symbol in pattern:
                for n in cur.next:
                    if n.val == symbol:
                        cur = n
                        nodes.append(cur)
                        hidden.append(cur.h)
                        break
                else:
                    break
            if len(hidden) == len(pattern):
                yield nodes, pattern, hidden, support


if __name__ == "__main__":
    from utils import RNNLoader

    loader = RNNLoader('synthetic_data_1', 'gru')

    pattern_sampler = PatternSampler(loader, absorb=False, pos_threshold=.95, sample_threshold=5, return_sample=True)
    for i, res in enumerate(pattern_sampler):
        _, p, _, _ = res
        print("Pattern %d: %s." % (i + 1, p))

    pass
