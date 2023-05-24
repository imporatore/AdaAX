import warnings

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, START_PREFIX, VOCAB_THRESHOLD, SYNTHETIC_DATA_DIR, TOMITA_DATA_DIR, REAL_DATA_DIR
from data.utils import load_pickle, load_csv
from RNN.Helper_Functions import preprocess, tokenize, build_vocab, pad_seq2idx, Vocab


# todo: synthetic doesn't need padding
class SyntheticDataset(Dataset):
    """ Known alphabet

    """
    def __init__(self, data, alphabet, start_prefix, pad_len, vocab=None):
        X, y = data
        pad_len, pad_symbol = pad_len + len(start_prefix), '<pad>'
        tokens = [start_prefix + list(expr) for expr in X]

        # build vocabulary, add start symbol and padding symbol
        if vocab:
            self.vocab = vocab
        else:
            self.vocab, self.alphabet = Vocab(), list(alphabet)
            for s in [pad_symbol] + start_prefix + self.alphabet:
                self.vocab.add_word(s)

        # pad to fix length & substitute with index
        # although 1 stands for unknown in func pad_seq2idx, as long as the alphabet is sufficient for the
        # synthetic data, there is no ambiguity
        seqs = pad_seq2idx(tokens, pad_len, self.vocab.word2idx)

        self.data = list(zip(seqs, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


class PolarityDataset(Dataset):
    """ Construct a dataset composed of polarity text and labels from csv.

    Attributes:
        df (Dataframe): Dataframe of the CSV from teh path
        vocab (dict{str: int}: A vocabulary dictionary from word to indices for this dataset
        samples_weight(ndarray, shape(len(labels),)): An array with each sample_weight[i] as the weight of the ith sample
        data (list[int, [int]]): The data in the set
    """

    def __init__(self, df, pad_len, min_count, vocab=None, start_prefix=None):
        self.df, pad_len = df, pad_len + len(start_prefix)

        self.df.loc[:, "text"] = self.df["text"].apply(preprocess).values  # preprocess
        self.df['words'] = self.df["text"].apply(tokenize)  # tokenize
        self.df.loc[:, 'words'] = self.df['words'].apply(lambda x: start_prefix + x).values
        self.df = self.df.loc[self.df['words'].apply(len) > 1].reset_index(drop=True)  # filter out empty rows

        self.vocab = build_vocab(self.df['words'], min_count, vocab)  # build vocab
        seqs = pad_seq2idx(self.df['words'], pad_len, self.vocab.word2idx)  # pad to fix length & substitute with index

        # check start symbol
        if start_prefix:
            assert start_prefix[0] in self.vocab.word2idx.keys(), \
                warnings.warn("Start symbol %s not in the vocabulary." % start_prefix[0])

        # compute sample weights from inverse class frequencies
        class_sample_count = np.unique(self.df['label'], return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight[self.df['label']])

        self.data = list(zip(seqs, self.df['label']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def get_loader(fname, batch_size):

    if fname in ["synthetic_data_1", "synthetic_data_2", "tomita_data_1", "tomita_data_2"]:
        ftype = 'synthetic'
    elif fname in ["yelp_review_balanced"]:
        ftype = 'real'
    else:
        raise ValueError('File %s not found.' % fname)

    if ftype == 'synthetic':

        if fname in ["synthetic_data_1", "synthetic_data_2"]:
            pad_len, data_dir = 15, SYNTHETIC_DATA_DIR
        else:
            pad_len, data_dir = 30, TOMITA_DATA_DIR

        X, y = load_pickle(data_dir, fname)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        alphabet = '01'
        train_dataset = SyntheticDataset((X_train, y_train), alphabet, start_prefix=START_PREFIX, pad_len=pad_len)
        test_dataset = SyntheticDataset((X_test, y_test), alphabet, START_PREFIX, pad_len, train_dataset.vocab)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        return train_dataloader, None, test_dataloader, train_dataset.vocab

    else:
        data = load_csv(REAL_DATA_DIR, fname)
        train_df = data.iloc[:int(data.shape[0] * .6)].reset_index(drop=True)
        valid_df = data.iloc[int(data.shape[0] * .6): int(data.shape[0] * .8)].reset_index(drop=True)
        test_df = data.iloc[int(data.shape[0] * .8):].reset_index(drop=True)

        if fname == "yelp_review_balanced":
            pad_len = 25

        train_dataset = PolarityDataset(train_df, pad_len, min_count=VOCAB_THRESHOLD, start_prefix=START_PREFIX)
        valid_dataset = PolarityDataset(valid_df, pad_len, VOCAB_THRESHOLD, train_dataset.vocab, START_PREFIX)
        test_dataset = PolarityDataset(test_df, pad_len, VOCAB_THRESHOLD, train_dataset.vocab, START_PREFIX)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(dataset=valid_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        return train_dataloader, valid_dataloader, test_dataloader, train_dataset.vocab
