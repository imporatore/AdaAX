from collections import defaultdict


class ForwardTransition(dict):

    def __init__(self, table, state, **kwargs):
        self.table, self.state = table, state
        super().__init__(**kwargs)

    def __getitem__(self, item):
        return super(ForwardTransition, self).__getitem__(item)

    def __setitem__(self, key, value):
        """ Update transition self.state ----> key ----> value."""
        if key in self.keys():
            self.table.backward[self[key]][key].remove(self.state)
        super(ForwardTransition, self).__setitem__(key, value)
        if self.state not in self.table.backward[value][key]:
            self.table.backward[value][key].append(self.state)

    def __missing__(self, key):
        # self[key] = None
        # return super(ForwardTransition, self).__getitem__(key)
        return None


class BackwardTransition(dict):

    def __init__(self, table, state, **kwargs):
        self.table, self.state = table, state
        super().__init__(**kwargs)

    def __getitem__(self, item):
        return super(BackwardTransition, self).__getitem__(item)

    def __missing__(self, key):
        self[key] = list()
        return super(BackwardTransition, self).__getitem__(key)


class Table(dict):

    def __init__(self, table, direction, **kwargs):
        self.table, self.direction = table, direction
        super().__init__(**kwargs)

    def __getitem__(self, item):
        return super(Table, self).__getitem__(item)

    def __missing__(self, key):
        if self.direction == 'forward':
            self[key] = ForwardTransition(self.table, key)
        elif self.direction == 'backward':
            self[key] = BackwardTransition(self.table, key)
        return super(Table, self).__getitem__(key)


class TransitionTable:

    def __init__(self):
        self.forward = Table(self, direction='forward')  # dict{state: dict{symbol: state}}
        self.backward = Table(self, direction='backward')  # dict{state: dict{symbol: list[state]}}

    def __getitem__(self, item):
        return self.forward[item]

    def __setitem__(self, key, value):
        self.forward.__setitem__(key, value)
        for symbol, state in value.items():
            self.backward[state][symbol].append(key)

    def keys(self):
        return self.forward.keys()

    def get(self, *args, **kwargs):
        return self.forward.get(*args, **kwargs)

    def get_backward(self, *args, **kwargs):
        return self.backward.get(*args, **kwargs)

    def pop(self, item):
        forward_transition = self.forward.pop(item)
        backward_transition = self.backward.pop(item)

        # remove entering transition in forward dict
        for symbol in backward_transition.keys():
            for state in backward_transition[symbol]:
                if state != item:  # self-loop
                    del self.forward[state][symbol]

        # remove exiting transition in backward dict
        for symbol, state in forward_transition.items():
            if state != item:  # self-loop
                self.backward[state][symbol].remove(item)
                if not self.backward[state][symbol]:
                    del self.backward[state][symbol]
                    if not self.backward[state]:
                        del self.backward[state]

        return forward_transition, backward_transition

    def _check_transition_consistency(self):
        """ Check consistency of forward and backward transitions."""
        forward_edges_dict, backward_edges_dict = defaultdict(set), defaultdict(set)

        for parent in self.forward.keys():
            for s in self.forward[parent].keys():
                child = self.forward[parent][s]
                forward_edges_dict[(parent, child)].add(s)

        for child in self.backward.keys():
            for s in self.backward[child].keys():
                for parent in self.backward[child][s]:
                    backward_edges_dict[(parent, child)].add(s)

        assert forward_edges_dict == backward_edges_dict, \
            "Forward transitions doesn't match backward transitions."

    def _check_state_consistency(self, states):
        """ Check if the states in transition table are consistent with given states."""
        assert all([s in states for s in self.forward.keys()]), "Forward table has unidentified parent state."
        assert all([s in states for s in self.backward.keys()]), "Backward table has unidentified child state."
        for state in self.forward.keys():
            assert all([s in states for s in self.forward[state].values()]), \
                "Forward table has unidentified child state."
        for state in self.backward.keys():
            for symbol in self.backward[state].keys():
                assert all([s in states for s in self.backward[state][symbol]]), \
                    "Backward table has unidentified parents state."

    def _check_empty_transition(self):
        """ Check if empty transition exists."""
        for state in self.forward.keys():
            assert not any([s is None for s in self.forward[state].values()]), \
                "Empty transition found in forward table."
        for state in self.backward.keys():
            assert not any([s == [] for s in self.forward[state].values()]), \
                "Empty transition found in backward table."


if __name__ == '__main__':
    import random

    delta = TransitionTable()
    for i in range(100):
        delta[random.randint(0, 10)][random.choice('abcdefgh')] = random.randint(0, 10)
    delta._check_transition_consistency()
    delta._check_state_consistency([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(delta.pop(1))
    delta._check_transition_consistency()
    delta._check_state_consistency([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    delta._check_state_consistency([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pass
