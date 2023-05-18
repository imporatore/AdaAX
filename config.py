import os

# RNN parameter
VOCAB_SIZE = 10000

# Data path
DATA_DIR = r'./data/'
RNN_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_rnn.npy')
LSTM_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_lstm.npy')
GRU_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_gru.npy')

# todo: add command line arguments
# todo: auto select cluster num
# Parameters:
# Cluster_id: 0, 1, 2, ... , C-1
K = 10  # Initial cluster numbers, determined by elbow method
THETA = 0.005  # Threshold for pruning
TAU = 1  # Threshold for neighbour distance
DELTA = 0.001  # Threshold for merging fidelity loss
