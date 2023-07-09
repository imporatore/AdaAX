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
TEMP_DIR = os.path.join(DATA_DIR, 'temp')

# ------------------------------ RNN parameter ------------------------------------
VOCAB_SIZE = 10000
HIDDEN_DIM = 64
VOCAB_THRESHOLD = 1
DROPOUT_RATE = .2
# Add START_SYMBOL for every input sequence of RNN to fetch the hidden values of h0 (for merging)
START_SYMBOL = '<START>'  # todo: modify the code for a None START_SYMBOL when we add no symbol to the input sequence
START_PREFIX = [START_SYMBOL] if START_SYMBOL else []

# ----------------------------- AdaAX parameter ------------------------------------
POS_THRESHOLD = .95  # threshold for an expression be considered a positive pattern
SAMPLE_THRESHOLD = 5  # sample threshold for positive patterns

TAU = 1.  # Threshold for neighbour distance
DELTA = 0.0001  # Threshold for merging fidelity loss
