# Ambiguity here.
# Hidden in core sets discovered during backtracking shares common suffix (and cluster label), not necessary the prefix.
# Hidden in core sets generated along 'extracted pattern', however, share common prefix, thence consistent hidden value.
# Core sets generated along 'extracted pattern' are called pure sets instead for disambiguation.

from utils import rnn_hidden_output


class PureSet:

    def __init__(self, prefix):
        # self._prefix = prefix
        self._prefix = [prefix]
        self._h = rnn_hidden_output(prefix)

    @property
    def prefixes(self):
        return self._prefix

    @property
    def initial_prefix(self):
        return self._prefix[0]

    # def _hidden_state_value(self):
    #     return rnn_hidden_output(self._prefix)


class State(PureSet):

    def __init__(self, prefix):
        # todo: the hidden state values is set to be a constant and never updates, even after merging.
        super().__init__(prefix)
