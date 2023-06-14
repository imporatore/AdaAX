class State:
    """ State of a DFA."""

    def __init__(self, hidden_values=None):
        """
        Args:
            hidden_values: None or float, hidden values of 'prefix' which initialize the State (as a PureSet)
        """
        # todo: hidden state value is set to be a constant and never updates, even after merging.
        self.h = hidden_values


def build_start_state():
    """ Build start state for the DFA.

    Returns:
        h0, a 'PureSet'(State) of start state.
    """
    return State()


if __name__ == "__main__":
    pass
