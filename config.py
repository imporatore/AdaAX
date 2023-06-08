import os


RANDOM_STATE = 13579

# ------------------ Path ------------------------
DATA_DIR = r"C:\PycharmProjects\AdaAX\data"

RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw_data')
SYNTHETIC_DATA_DIR = os.path.join(DATA_DIR, 'synthetic_data')
TOMITA_DATA_DIR = os.path.join(DATA_DIR, 'tomita_data')
REAL_DATA_DIR = os.path.join(DATA_DIR, 'real_data')

VOCAB_DIR = os.path.join(DATA_DIR, 'vocab')
DATALOADER_DIR = os.path.join(DATA_DIR, 'dataloader')
RNN_MODEL_DIR = os.path.join(DATA_DIR, 'rnn_checkpoints')
RNN_RESULT_DIR = os.path.join(DATA_DIR, 'rnn_result')

DFA_DIR = os.path.join(DATA_DIR, 'dfa')
IMAGE_DIR = os.path.join(DATA_DIR, 'image')

# ------------------------------ RNN parameter ------------------------------------
VOCAB_SIZE = 10000
HIDDEN_DIM = 64
VOCAB_THRESHOLD = 1
DROPOUT_RATE = .2
# Add START_SYMBOL for every input sequence of RNN to fetch the hidden values of h0 (for merging)
START_SYMBOL = '<START>'  # todo: modify the code for a None START_SYMBOL when we add no symbol to the input sequence
START_PREFIX = [START_SYMBOL] if START_SYMBOL else []

# ----------------------------- AdaAX parameter ------------------------------------
# todo: add command line arguments
# todo: auto select cluster num
# Parameters:
# Cluster_id: -1, 0, 1, 2, ... , C-1 (-1 is start state)
K = 10  # Initial cluster numbers, determined by elbow method
# THETA = 0.005  # Threshold for pruning
THETA = 0.
TAU = 1.  # Threshold for neighbour distance
DELTA = 0.001  # Threshold for merging fidelity loss

# Plot
SEP = ", "
