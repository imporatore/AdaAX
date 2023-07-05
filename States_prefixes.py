class State:
    """ State (in DFA) is a set of 'PureSets'(prefixes).

    Holds the prefixes of all PureSets this state contains.
    """

    def __init__(self, prefix: list, hidden_values=None, weight=0.):
        """
        Args:
            prefix: list, a list of symbols which initialize the State (as a PureSet).
                - the first prefix upon which the PureSet is built.
                - also, the hidden value of this PureSet is evaluated on this prefix.
            hidden_values: None or float, hidden values of 'prefix'
            weight: float, weight for updating merged state's hidden values
        """
        # todo: hidden state value is set to be a constant and never updates, even after merging.
        if not isinstance(prefix, list):
            raise ValueError("Prefix must be a list.")
        self.prefixes = [prefix]
        self.h = hidden_values
        self.weight = weight


def build_start_state():
    """ Build start state for the DFA.

    Returns:
        h0, a 'PureSet'(State) of start state.
    """
    return State([], weight=1.)


if __name__ == "__main__":
    pass
