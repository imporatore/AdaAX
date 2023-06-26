import warnings
from collections import defaultdict


class WrappedDict:
    """ Helper class for a wrapped dict.

    So that a class may have additional attributes and methods while still behaves like a dict.
    """

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

    def todict(self):
        return self._items

    def __missing__(self, key):
        pass


class ForwardTransition(WrappedDict):
    """ Forward transition table for one state.

    Backward transitions would be updated automatically after forward transitions changes.
    """

    def __init__(self, table, state, *args, **kwargs):
        """
        Args:
            table: TransitionTable
            state: the start state of the forward transitions
        """
        self.__table, self.__state = table, state
        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        """ Update transition self.state ----> key ----> value."""
        if key in self._items.keys():
            # remove existing backward transition
            self.__table.backward[self._items[key]][key].remove(self.__state)
        self._items.__setitem__(key, value)
        # update backward transition
        if self.__state not in self.__table.backward[value][key]:
            self.__table.backward[value][key].add(self.__state)

    def __missing__(self, key):
        """ Called when key missing in self._items[key]. Return None."""
        return None


class BackwardTransition(WrappedDict):
    """ Backward transition table for one state."""

    def __init__(self, table, state, *args, **kwargs):
        """
        Args:
            table: TransitionTable
            state: the start state of the backward transitions
        """
        self.__table, self.__state = table, state
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        """ Called when key missing in self._items[key]. Set the value (parents) to an empty list [] and return."""
        self._items[key] = set()
        return self._items.__getitem__(key)

    def __len__(self):
        """ Count of all backward transitions of the state self.__state."""
        return sum([len(transit) for transit in self.values()])


class Table(WrappedDict):
    """ Transition table for DFA in one direction."""

    def __init__(self, table, direction, *args, **kwargs):
        """
        Args:
            table: TransitionTable
            direction: str, choices: ['forward', 'backward']
        """
        self.__table, self.__direction = table, direction
        super().__init__(*args, **kwargs)

    def __missing__(self, key):
        """ Called when key (state) missing in self._items[key]. Set an empty transition for the state and return."""
        if self.__direction == 'forward':
            self._items[key] = ForwardTransition(self.__table, key)
        elif self.__direction == 'backward':
            self._items[key] = BackwardTransition(self.__table, key)
        return self._items.__getitem__(key)

    def __len__(self):
        """ Count of all self.__direction ('forward'/'backward') transitions."""
        return sum([len(transition) for transition in self.values()])

    def pop(self, k, **kwargs):
        """ Pop the transition table for one state.

        When the state doesn't have its transition table, return an empty table."""
        try:
            return self._items.pop(k, **kwargs)
        except KeyError:
            if self.__direction == 'forward':
                return ForwardTransition(self.__table, k)
            elif self.__direction == 'backward':
                return BackwardTransition(self.__table, k)


class TransitionTable:  # todo: __new__
    """ Bidirectional transition table.

    Most methods act upon forward table."""

    def __init__(self):
        self.forward = Table(self, direction='forward')  # dict[state: dict[symbol: state]]
        self.backward = Table(self, direction='backward')  # dict[state: dict[symbol: list[state]]]

    def __getitem__(self, item):
        """ Get item from forward table."""
        return self.forward[item]

    def __setitem__(self, key, value):
        """ Directly set a transition dict of a state.

        Deprecated."""
        self.forward.__setitem__(key, value)
        for symbol, state in value.items():
            self.backward[state][symbol].add(key)

    def __len__(self):
        """ Count of all forward transitions (the num of backward transitions are the same)."""
        return self.forward.__len__()

    def keys(self):
        """ States in the forward transition table."""
        return self.forward.keys()

    def values(self):
        """ Transitions in the forward transition table."""
        return self.forward.values()

    def items(self):
        """ Items in the forward transition table."""
        return self.forward.items()

    def get(self, *args, **kwargs):
        """ 'Get' method of the forward transition table."""
        return self.forward.get(*args, **kwargs)

    def get_backward(self, *args, **kwargs):
        """ 'Get' method of the backward transition table."""
        return self.backward.get(*args, **kwargs)

    def pop(self, item):
        """ Pop a state out of the DFA. Remove all the relevant forward & backward transitions.

        Return:
            forward_transition, backward_transition: tuple(dict, dict),
                - set the output into dict to prevent the changing of popped transitions from affecting
                existing transition table
        """
        forward_transition = self.forward.pop(item)
        backward_transition = self.backward.pop(item)

        # remove entering transition in forward dict
        for symbol in backward_transition.keys():
            for state in backward_transition[symbol]:
                if state != item:  # self-loop, forward transition already popped
                    del self.forward[state][symbol]

        # remove exiting transition in backward dict
        for symbol, state in forward_transition.items():
            if state != item:  # self-loop, backward transition already popped
                self.backward[state][symbol].remove(item)

                # check empty transitions
                if not self.backward[state][symbol]:
                    del self.backward[state][symbol]
                    if not self.backward[state]:
                        del self.backward[state]

        return forward_transition.todict(), backward_transition.todict()

    def delete_forward(self, item):
        """ Delete the forward transitions of a state, i.e., for the accept states of absorb=True DFA."""
        forward_transition = self.forward.pop(item)

        # remove exiting transition in backward dict
        for symbol, state in forward_transition.items():
            self.backward[state][symbol].remove(item)

            # check empty transitions
            if not self.backward[state][symbol]:
                del self.backward[state][symbol]
                if not self.backward[state]:
                    del self.backward[state]

    def copy_by_mapping(self, mapping):
        new_table = TransitionTable()
        for state in self.forward.keys():
            for s in self.forward[state].keys():
                new_table[mapping[state]][s] = mapping[self.forward[state][s]]
        return new_table

    def _check_transition_consistency(self):
        """ Check consistency of forward and backward transitions by edges.

        Note:
            edges is a dict: dict[(parent, child)] = [transition_symbol1, transition_symbol2, ...]
        """
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
        to_delete_forward_transitions, to_delete_backward_transitions = [], []
        to_delete_forward_states, to_delete_backward_states = [], []

        for state in self.forward.keys():
            for s in self.forward[state].keys():
                if self.forward[state][s] is None:
                    warnings.warn("Empty transition found in forward table.")
                    to_delete_forward_transitions.append((state, s))

            if len(self.forward[state]) == 0:
                warnings.warn("Empty forward transition table found.")
                to_delete_forward_states.append(state)

        for state, s in to_delete_forward_transitions:
            del self.forward[state][s]

        for state in to_delete_forward_states:
            del self.forward[state]

        for state in self.backward.keys():
            for s in self.backward[state].keys():
                if not self.backward[state][s]:
                    warnings.warn("Empty transition found in backward table.")
                    to_delete_backward_transitions.append((state, s))

            if len(self.backward[state]) == 0:
                warnings.warn("Empty backward transition table found.")
                to_delete_backward_states.append(state)

        for state, s in to_delete_backward_transitions:
            del self.backward[state][s]

        for state in to_delete_backward_states:
            del self.backward[state]


if __name__ == '__main__':
    pass
