import os

# Data path
DATA_DIR = r"D:\PycharmProjects\AdaAX\data"

RNN_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_rnn.npy')
LSTM_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_lstm.npy')
GRU_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_gru.npy')
