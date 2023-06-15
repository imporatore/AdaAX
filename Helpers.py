from functools import wraps


def substitute(lst: list, element1, element2):
    """ Substitute element1 with element2 in list lst."""
    if element1 in lst:
        lst.remove(element1)
        if element2 not in lst:
            lst.append(element2)


# todo: examine the order of these ops
def check_consistency(dfa, check_transition=True, check_state=True, check_empty=True, check_null_states=True):
    """ Consistency check of DFA.

    Args:
        check_transition: bool, default=True, check if forward and backward transitions are consistent
        check_state: bool, default=True, check if there are unrecognized states in forward transitions
        check_empty: bool, default=True, check if there is empty(null)
            children state in transitions (due to usage of defaultdict)
        check_null_states: bool, default=True, check if there are unreachable states
    """

    dfa._check_absorbing()  # check if accept states have exiting transitions for DFA which absorb=True

    try:
        dfa._check_accept_states()
    except AssertionError as message:
        raise RuntimeError(message)

    if check_null_states:
        dfa._check_null_states()

    if check_transition:
        try:
            dfa._check_transition_consistency()
        except AssertionError as message:
            raise RuntimeError(message)

    if check_state:
        try:
            dfa._check_state_consistency()
        except AssertionError as message:
            raise RuntimeError(message)

    if check_empty:
        dfa._check_empty_transition()


class ConsistencyCheck:
    """ Decorator of checks of DFA operations.

    Usage:
        dfa_check = ConsistencyCheck(dfa)

        @dfa_check(check_transition=True, check_state=True, check_empty=False, check_null_states=True)
        def operation(dfa):
            pass
    """

    def __init__(self, dfa):
        self.dfa = dfa

    def __call__(self, check_transition=True, check_state=True, check_empty=False, check_null_states=True):

        def _consistency_check(func):
            @wraps(func)
            def __wrapper(*args, **kwargs):
                res = func(*args, **kwargs)
                check_consistency(self.dfa, check_transition, check_state, check_empty, check_null_states)
                return res

            return __wrapper

        return _consistency_check
