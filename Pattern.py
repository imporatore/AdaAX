from collections import deque


class PatternSampler:

    def __init__(self, rnn_loader, pos_threshold):
        """
        Args:
            rnn_loader: RNNLoader
            pos_threshold: float, threshold for positive patterns, default 0.95,
                i.e. (pos_sup / (pos_sup + neg_sup)) > pos_threshold
                - should be adjusted for unbalanced dataset
        """
        self.root = rnn_loader.prefix_tree.root
        self.threshold = pos_threshold

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
            expr: list, list of symbols, positive patterns
            hidden: list, list of hidden values in expr
            sup: tuple(float, float), positive and negative support of the pattern
        """
        queue = deque([(self.root, [], [])])
        while queue:
            node, expr, hidden = queue.popleft()

            if node.pos_sup / (node.pos_sup + node.neg_sup) >= self.threshold:
                if node.val == '<pad>':  # only negative patterned
                    assert expr[-2] != '<pad>'
                    yield expr[:-1], hidden[:-1], (node.pos_sup, node.neg_sup)  # todo: sup incorrect
                else:  # positive patterned or large positive support when negative patterned
                    yield expr, hidden, (node.pos_sup, node.neg_sup)

            else:  # todo: check (seems correct for both positive & negative patterns cases)
                if node.val != '<pad>':  # negative sample when both positive patterned & negative patterned
                    for n in node.next:
                        queue.append((n, expr + [n.val], hidden + [n.h]))


class PatternIterator:
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
            pattern: list, list of symbols, external patterns which can be found in the rnn_loader
            hidden: list, list of hidden values of the pattern
            support: tuple(float, float), positive and negative support of the pattern
        """
        for pattern, support in zip(self.patterns, self.support):
            cur, hidden = self.root, []
            for symbol in pattern:
                for n in cur.next:
                    if n.val == symbol:
                        cur = n
                        hidden.append(cur.h)
                        break
                else:
                    break
            if len(hidden) == len(pattern):
                yield pattern, hidden, support


if __name__ == "__main__":
    from utils import RNNLoader

    loader = RNNLoader('synthetic_data_1', 'gru')

    pattern_sampler = PatternSampler(loader, pos_threshold=.95)
    for p, _, sup in pattern_sampler:
        print(p, sup)

    pass
