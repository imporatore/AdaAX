import numpy as np


class RNNLoader:

    def __int__(self, alphabet, rnn_data):
        """ Result loader of trained RNN for extracting patterns and building DFA.

        Also maintain function of calculating DFA fidelity.

        Args:
            - alphabet: list of shape (VOCAB_SIZE)
            - rnn_data: (input_sequence, hidden_states, rnn_output)
                - input_sequence, np.array of shape (N, PAD_LEN)
                - hidden_states, np.array of shape (N, PAD_LEN, hidden_dim)
                - rnn_output, np.array of shape (N,)
        """
        self.alphabet = alphabet
        self.input_sequence, self.hidden_states, self.rnn_output = rnn_data

        # Check shape
        assert self.input_sequence.shape[0] == self.hidden_states.shape[0]
        assert self.input_sequence.shape[1] == self.hidden_states.shape[1]
        assert self.input_sequence.shape[0] == self.rnn_output.shape[0]

    def eval_fidelity(self, dfa):
        """ Evaluate the fidelity of (extracted) DFA."""
        return np.mean([dfa.classify_expression(expr) == ro for expr, ro in zip(self.input_sequence, self.rnn_output)])

    # The hidden value in the RNN for given prefix
    # todo: Accelerate by using cashed hidden states
    # todo: should move into class RNNLoader
    def rnn_hidden_values(self, prefix):
        pass


def d(hidden1: np.array, hidden2: np.array):
    """ Euclidean distance of hidden state values."""
    return np.sqrt(np.sum((hidden1 - hidden2) ** 2))


# todo: add logger


if __name__ == "__main__":
    pass
