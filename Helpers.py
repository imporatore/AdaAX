from functools import wraps


# todo: examine the order of these ops
def check_consistency(dfa, check_transition=True, check_state=True, check_empty=True, check_null_states=True):
    if check_transition:  # only available in bidirectional transition table
        try:
            dfa.delta._check_transition_consistency()
        except AssertionError as message:
            raise RuntimeError(message)

    if check_state:
        try:
            dfa.delta._check_state_consistency(dfa.Q)
        except AssertionError as message:
            raise RuntimeError(message)

    if check_empty:
        dfa.delta._check_empty_transition()

    if check_null_states:
        dfa._check_null_states()


class ConsistencyCheck:

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
