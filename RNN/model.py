import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class VanillaRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_rate, vocab):
        super(VanillaRNN, self).__init__()

        input_size, output_size = len(vocab), 1

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.RNN(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        rnn_out, hidden_state = self.rnn(packed)
        # out = self.linear(rnn_out[:])
        out = self.linear(hidden_state.squeeze(0))
        return out


class VanillaLSTMModel(nn.Module):

    def __init__(self, embedding_size, hidden_size, dropout_rate, vocab):
        super(VanillaLSTMModel, self).__init__()

        input_size, output_size = len(vocab), 1

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        # lstm_out: tensor containing all output hidden states, for each timestep. shape: (length, batch, hidden_size)
        # hidden_state: tensor containing the hidden state for last timestep. shape: (1, batch, hidden_size)
        # cell state: tensor containing the cell state for last timestep. shape: (1, batch, hidden_size)
        lstm_out, (hidden_state, cell_state) = self.lstm(packed)
        out = self.linear(hidden_state.squeeze(0))
        return out


class VanillaGRUModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout_rate, vocab):
        super(VanillaGRUModel, self).__init__()

        input_size, output_size = len(vocab), 1

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.gru = nn.GRU(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        gru_out, hidden_state = self.gru(packed)
        out = self.linear(hidden_state.squeeze(0))
        return out


class GloveModel(nn.Module):
    """
    Pretrained Embedding + Pooling of LSTM Hidden States
    """

    def __init__(self, embedding_size, hidden_size, dropout_rate, glove):
        super(GloveModel, self).__init__()

        output_size = 1

        self.embedding = nn.Embedding.from_pretrained(glove)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(3 * hidden_size, output_size)

    def forward(self, input, lengths):
        embeddings = self.embedding(input)
        embeddings = self.dropout(embeddings)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)

        lstm_out, (hidden_state, cell_state) = self.lstm(packed, num_layers=2)
        lstm_out, lengths = pad_packed_sequence(lstm_out)

        # pool the lengths
        avg_pool = F.adaptive_avg_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()
        max_pool = F.adaptive_max_pool1d(lstm_out.permute((1, 2, 0)), 1).squeeze()

        # concat forward and pooled states
        concat = torch.cat((hidden_state[-1, :, :], max_pool, avg_pool), dim=1)
        out = self.linear(concat)
        return out


if __name__ == '__main__':
    pass
