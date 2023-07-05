import numpy as np


class State:
    """ State of a DFA."""

    def __init__(self, hidden_values, weight=0.):
        """
        Args:
            hidden_values: float, hidden values of 'prefix' which initialize the State (as a PureSet)
            weight: float, weight for updating merged state's hidden values
        """
        self.h = hidden_values
        self.weight = weight


# todo: weight only positive support
def build_start_state(loader):
    """ Build start state for the DFA.

    Args:
        loader: RNNLoader

    Returns:
        h0, a 'PureSet'(State) of start state.
    """
    hidden_dim = loader.hidden_states.shape[-1]
    return State(hidden_values=np.zeros(hidden_dim), weight=loader.prefix_tree.root.pos_sup)


if __name__ == "__main__":
    pass
