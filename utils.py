import numpy as np


class RNNLoader:

    def __int__(self, alphabet, rnn_data):
        """

        Param:
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
        return np.mean([dfa.classify_expression(expr) == ro for expr, ro in zip(self.input_sequence, self.rnn_output)])


# The hidden value in the RNN for given prefix
# todo: Accelerate by using cashed hidden states
def rnn_hidden_values(prefix):
    pass


# Euclidean distance of hidden state values
def d(hidden1, hidden2):
    return np.sqrt(np.sum((hidden1 - hidden2) ** 2))


if __name__ == "__main__":
    pass
