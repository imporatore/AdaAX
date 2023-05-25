import warnings
import os

import torch
import numpy as np

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, VOCAB_THRESHOLD, SYNTHETIC_DATA_DIR, TOMITA_DATA_DIR, REAL_DATA_DIR, VOCAB_DIR, DATALOADER_DIR
from data.utils import load_pickle, load_csv, save2pickle
from RNN.Helper_Functions import preprocess, tokenize, build_vocab, pad_seq2idx, Vocab


# todo: synthetic data doesn't need padding
class SyntheticDataset(Dataset):
    """ Synthetic dataset for known alphabet."""
    
    def __init__(self, data, alphabet, start_prefix, pad_len, vocab=None, pad=True):
        X, y = data
        pad_len = pad_len + len(start_prefix)
        pad_ = ['<pad>'] if pad else []
        tokens = [start_prefix + list(expr) for expr in X]

        if vocab:
            self.vocab = vocab
        else:
            # build vocabulary, add start symbol and padding symbol
            self.vocab, self.alphabet = Vocab(), list(alphabet)
            for s in pad_ + start_prefix + self.alphabet:
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


def get_loader(fname, batch_size, start_symbol, load_vocab, save_vocab, load_loader, save_loader):

    if load_loader:
        try:
            return load_pickle(DATALOADER_DIR, fname)
        except FileNotFoundError:
            pass

    loaded_vocab = None

    if load_vocab:
        try:
            loaded_vocab = load_pickle(VOCAB_DIR, fname)
        except FileNotFoundError:
            pass

    start_prefix = [start_symbol] if start_symbol else []

    if fname in ["synthetic_data_1", "synthetic_data_2", "tomita_data_1", "tomita_data_2"]:
        ftype = 'synthetic'
    elif fname in ["yelp_review_balanced"]:
        ftype = 'real'
    else:
        raise ValueError('File %s not found.' % fname)

    if ftype == 'synthetic':

        if fname in ["synthetic_data_1", "synthetic_data_2"]:
            pad_len, data_dir, pad = 15, SYNTHETIC_DATA_DIR, False
        else:
            pad_len, data_dir, pad = 30, TOMITA_DATA_DIR, True

        X, y = load_pickle(data_dir, fname)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

        alphabet = '01'
        train_dataset = SyntheticDataset((X_train, y_train), alphabet, start_prefix, pad_len, loaded_vocab, pad)
        test_dataset = SyntheticDataset((X_test, y_test), alphabet, start_prefix, pad_len, train_dataset.vocab, pad)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_dataloader = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        dataloaders = (train_dataloader, None, test_dataloader, train_dataset.vocab)

    else:
        data = load_csv(REAL_DATA_DIR, fname)
        train_df = data.iloc[:int(data.shape[0] * .6)].reset_index(drop=True)
        valid_df = data.iloc[int(data.shape[0] * .6): int(data.shape[0] * .8)].reset_index(drop=True)
        test_df = data.iloc[int(data.shape[0] * .8):].reset_index(drop=True)

        if fname == "yelp_review_balanced":
            pad_len = 25

        train_dataset = PolarityDataset(train_df, pad_len, VOCAB_THRESHOLD, loaded_vocab, start_prefix)
        valid_dataset = PolarityDataset(valid_df, pad_len, VOCAB_THRESHOLD, train_dataset.vocab, start_prefix)
        test_dataset = PolarityDataset(test_df, pad_len, VOCAB_THRESHOLD, train_dataset.vocab, start_prefix)

        train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_dataloader = DataLoader(dataset=valid_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)
        test_dataloader = DataLoader(dataset=test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        dataloaders = (train_dataloader, valid_dataloader, test_dataloader, train_dataset.vocab)

    if (loaded_vocab is None) and save_vocab:
        save2pickle(VOCAB_DIR, train_dataset.vocab, fname)

    if save_loader:
        save2pickle(DATALOADER_DIR, dataloaders, fname)

    return dataloaders
