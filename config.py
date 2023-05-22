import os

# RNN parameter
VOCAB_SIZE = 10000
HIDDEN_DIM = 64
# Add START_SYMBOL for every input sequence of RNN to fetch the hidden values of h0 (for merging)
START_SYMBOL = '<START>'  # todo: modify the code for a None START_SYMBOL when we add no symbol to the input sequence
START_PREFIX = [START_SYMBOL] if START_SYMBOL else []

# Data path
DATA_DIR = r"D:\PycharmProjects\AdaAX\data"
RNN_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_rnn.npy')
LSTM_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_lstm.npy')
GRU_HIDDEN_DIR = os.path.join(DATA_DIR, 'hidden_params_gru.npy')
RANDOM_STATE = 13579

# todo: add command line arguments
# todo: auto select cluster num
# Parameters:
# Cluster_id: -1, 0, 1, 2, ... , C-1 (-1 is start state)
K = 10  # Initial cluster numbers, determined by elbow method
THETA = 0.005  # Threshold for pruning
TAU = 1  # Threshold for neighbour distance
DELTA = 0.001  # Threshold for merging fidelity loss

# Plot
SEP = ", "
