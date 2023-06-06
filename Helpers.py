from functools import wraps


def check_consistency(dfa, check_transition=True, check_state=True, check_empty=False):
    if check_transition:
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
        try:
            dfa.delta._check_empty_transition()
        except AssertionError as message:
            raise RuntimeWarning(message)


class ConsistencyCheck:

    def __init__(self, dfa):
        self.dfa = dfa

    def __call__(self, check_transition=True, check_state=True, check_empty=False):

        def _consistency_check(func):
            @wraps(func)
            def __wrapper(*args, **kwargs):
                res = func(*args, **kwargs)
                check_consistency(self.dfa, check_transition, check_state, check_empty)
                return res

            return __wrapper

        return _consistency_check
