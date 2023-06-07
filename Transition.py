import copy
from collections import defaultdict


class WrappedDict:

    def __init__(self, *args, **kwargs):
        self._items = dict(*args, **kwargs)

    def __contains__(self, item):
        return self._items.__contains__(item)

    def __iter__(self):
        return self._items.__iter__()

    def __len__(self):
        return self._items.__len__()

    def __bool__(self):
        return bool(self._items)

    def __getitem__(self, item):
        if item in self._items.keys():
            return self._items.__getitem__(item)
        else:
            return self.__missing__(item)

    def __setitem__(self, key, value):
        self._items.__setitem__(key, value)

    def __delitem__(self, key):
        self._items.__delitem__(key)

    def keys(self):
        return self._items.keys()

    def values(self):
        return self._items.values()

    def items(self):
        return self._items.items()

    def get(self, *args, **kwargs):
        return self._items.get(*args, **kwargs)

    def pop(self, k, **kwargs):
        return self._items.pop(k, **kwargs)

    def popitem(self):
        return self._items.popitem()

    def update(self, *args, **kwargs):
        self._items.update(*args, **kwargs)

    def __missing__(self, key):
        pass


class ForwardTransition(WrappedDict):

    def __init__(self, table, state, *args, **kwargs):
        self.__table, self.__state = table, state
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        """ Update transition self.state ----> key ----> value."""
        if key in self._items.keys():
            self.__table.backward[self._items[key]][key].remove(self.__state)
        self._items.__setitem__(key, value)
        if self.__state not in self.__table.backward[value][key]:
            self.__table.backward[value][key].append(self.__state)

    def __missing__(self, key):
        # self._items[key] = None
        # return super(ForwardTransition, self).__getitem__(key)
        return None


class BackwardTransition(WrappedDict):

    def __init__(self, table, state, *args, **kwargs):
        self.__table, self.__state = table, state
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        self._items[key] = list()
        return self._items.__getitem__(key)

    def __len__(self):
        return sum([len(transit) for transit in self.values()])


class Table(WrappedDict):

    def __init__(self, table, direction, *args, **kwargs):
        self.__table, self.__direction = table, direction
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        if self.__direction == 'forward':
            self._items[key] = ForwardTransition(self.__table, key)
        elif self.__direction == 'backward':
            self._items[key] = BackwardTransition(self.__table, key)
        return self._items.__getitem__(key)

    def __len__(self):
        return sum([len(transition) for transition in self.values()])

    def pop(self, k, **kwargs):
        try:
            return self._items.pop(k, **kwargs)
        except KeyError:
            if self.__direction == 'forward':
                return ForwardTransition(self.__table, k)
            elif self.__direction == 'backward':
                return BackwardTransition(self.__table, k)


class TransitionTable:  # todo: __new__

    def __init__(self):
        self.forward = Table(self, direction='forward')  # dict{state: dict{symbol: state}}
        self.backward = Table(self, direction='backward')  # dict{state: dict{symbol: list[state]}}

    def __getitem__(self, item):
        return self.forward[item]

    def __setitem__(self, key, value):
        self.forward.__setitem__(key, value)
        for symbol, state in value.items():
            self.backward[state][symbol].append(key)

    def __len__(self):
        return self.forward.__len__()

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
    delta[1]['a'] = 2
    # delta.backward._Table__table
    # delta.forward._Table__table
    delta[1]['b'] = 3
    delta[2]['a'] = 3
    #
    # for i in range(100):
    #     delta[random.randint(0, 10)][random.choice('abcdefgh')] = random.randint(0, 10)
    # new_delta = copy.deepcopy(delta)
    # delta._check_transition_consistency()
    # delta._check_state_consistency([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    print(delta.pop(1))
    # delta._check_transition_consistency()
    # delta._check_state_consistency([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # delta._check_state_consistency([0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pass
