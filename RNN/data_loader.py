from __future__ import absolute_import

import warnings

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from ..config import RANDOM_STATE, START_PREFIX
from data.utils import load_pickle, load_csv
from RNN.config import VOCAB_THRESHOLD
from RNN.Helper_Functions import preprocess, tokenize, build_vocab, pad_seq2idx, Vocab


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

        self.data = list(zip(y, seqs))

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

        df["text"] = df["text"].progress_apply(preprocess)  # preprocess
        df['words'] = start_prefix + df["text"].progress_apply(tokenize)  # tokenize
        df['lengths'] = df['words'].apply(lambda x: min(len(x), pad_len))  # take lengths of words
        df = df.loc[df['lengths'] >= 1].reset_index(drop=True)  # filter out rows with lengths of 0

        self.vocab = build_vocab(df['words'], min_count, vocab)  # build vocab
        seqs = pad_seq2idx(df['words'], pad_len, self.vocab.word2idx)  # pad to fix length & substitute with index

        # check start symbol
        if start_prefix:
            assert start_prefix[0] in self.vocab.word2idx.keys(), \
                warnings.warn("Start symbol %s not in the vocabulary." % start_prefix[0])

        # compute sample weights from inverse class frequencies
        class_sample_count = np.unique(df['label'], return_counts=True)[1]
        weight = 1. / class_sample_count
        self.samples_weight = torch.from_numpy(weight[df['label']])

        self.data = list(zip(df['label'], seqs, df["lengths"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def get_loader(data_dir, fname, batch_size):

    if fname in ["synthetic_data_1", "synthetic_data_2", "tomita_data_1", "tomita_data_2"]:
        ftype = 'synthetic'
    elif fname in ["yelp_review_balanced"]:
        ftype = 'real'
    else:
        raise ValueError('File %s not found in %s.' % (fname, data_dir))

    if ftype == 'synthetic':
        X, y = load_pickle(data_dir, fname)

        if fname in ["synthetic_data_1", "synthetic_data_2"]:
            pad_len = 15
        else:
            pad_len = 30

        alphabet = '01'

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
        train_dataset = SyntheticDataset((X_train, y_train), alphabet, start_prefix=START_PREFIX, pad_len=pad_len)
        test_dataset = SyntheticDataset((X_test, y_test), alphabet, START_PREFIX, pad_len, train_dataset.vocab)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        return train_dataloader, test_dataloader, train_dataset.vocab

    else:
        data = load_csv(data_dir, fname)
        train_df, test_df = data.iloc[:int(data.shape[0] * .6)], data.iloc[int(data.shape[0] * .8):]
        valid_df = data.iloc[int(data.shape[0] * .6): int(data.shape[0] * .8)]

        if fname == "yelp_review_balanced":
            pad_len = 25

        train_dataset = PolarityDataset(train_df, pad_len, min_count=VOCAB_THRESHOLD, start_prefix=START_PREFIX)
        valid_dataset = PolarityDataset(valid_df, pad_len, VOCAB_THRESHOLD, train_dataset.vocab, START_PREFIX)
        test_dataset = PolarityDataset(train_df, pad_len, VOCAB_THRESHOLD, train_dataset.vocab, START_PREFIX)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(dataset=valid_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        return train_dataloader, valid_dataloader, test_dataloader, train_dataset.vocab