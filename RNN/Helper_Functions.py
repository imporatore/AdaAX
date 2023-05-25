import string
import re
from collections import Counter

import torch
import nltk
import numpy as np
from torchtext import vocab

stop_words = set(nltk.corpus.stopwords.words('english') + list(string.punctuation))


def preprocess(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'([/#\n])', r' \1 ', text)  # Add spaces around / and #
    text = re.sub(' {2,}', ' ', text)  # Remove extra spaces
    text = re.sub(r'(.)\1+', r'\1\1', text)  # Removes any repeated characters > 2 to 2
    text = re.sub(r'\w*\d\w*', '', text).strip()  # Remove any numbers and words mixed within them
    return text.replace("'s", "").replace("-", "")  # Remove 's -


# todo: use torchtext.data.utils.get_tokenizer instead
def tokenize(text):
    """ Return a tokenized word list for given text.

    Args:
        text: str, a doc with multiple sentences


    Returns:
        tokens: list, word list

    Example:
        Input: 'It is a nice day. I am happy.'
        Output: ['it', 'is', 'a', 'nice', 'day', 'i', 'am', 'happy']
    """
    tokens = []
    for word in nltk.casual_tokenize(text, preserve_case=False):
        if word not in stop_words and not word.isnumeric():
            tokens.append(word)
    return tokens


# todo: for given alphabet (synthetic dataset), <unk> is unnecessary
def build_vocab(sentence_list, min_count=0, vocab=None):
    """ Build vocabulary on given sentence list.

    Args:
        sentence_list: iterable object, an iterable object with multiple words in each sub-list
        vocab

    Params:
        min_count:int, minimum number of a word's count to be added into the vocabulary

    Returns:
        vocab: a dictionary from words to indices and indices to words
    """
    if vocab:
        return vocab

    counter = Counter()
    for sentence in sentence_list:
        counter.update(sentence)

    # sort by most common
    word_count = sorted(counter.items(), key=lambda x: x[1], reverse=True)

    # exclude words that are below a frequency
    words = [word for word, count in word_count if count > min_count]

    if vocab is None:
        vocab = Vocab()
        vocab.add_word('<pad>')  # 0 means the padding signal
        vocab.add_word('<unk>')  # 1 means the unknown word

    # add the words to the vocab
    for word in words:
        vocab.add_word(word)

    return vocab


def pad_seq2idx(data, pad_len, vocab_dict):
    """ Padding sequence and substitude words with their index in the vocabulary.

    Args:
        data: list, list of words
        pad_len: int, padding length of sequences
        vocab_dict: dict, vocabulary dict from words to indices

    Returns
        data_matrix: a dense sequence matrix whose elements are indices of words
    """
    data_matrix = np.zeros((len(data), pad_len), dtype=int)
    for i, doc in enumerate(data):
        for j, word in enumerate(doc):
            # todo
            if j == pad_len:
                break
            data_matrix[i, j] = vocab_dict.get(word, 1)  # 1 means the unknown word
    return data_matrix


class Vocab(object):
    """ Helper class for vocabulary dict."""

    def __init__(self):
        self.word2idx = dict()  # word to index dict
        self.words = []  # index to word
        self.vocab_size = 0  # size of vocabulary dict

    def __len__(self):
        return len(self.word2idx)

    def add_word(self, word):
        if word not in self.word2idx:  # update new word
            self.word2idx[word] = self.vocab_size
            self.words.append(word)
            self.vocab_size += 1

    def decode(self, seq):
        text = [self.words[i] for i in seq]
        text = [word for word in text if word != '<pad>']
        return ' '.join(text)

    def get_pretrained_embedding(self, name, embedding_dim):
        """ Fetch pretrained embeddings when building vocab.

        Currently available pretrained embeddings: 'glove' & 'fasttext'.
        Currently not in use.

        Args:
            name: str, choice: ['glove', 'fasttext'], type of pretrained embeddings used.
            embedding_dim: int, embedding dimension.

        Return
            embed: torch.tensor, pretrained embeddings fetched.Words not in the pretrained embeddings are initialized randomly.
        """
        if name == 'glove':
            pretrained_type = vocab.GloVe(name='42B', dim=embedding_dim)
        elif name == 'fasttext':
            if embedding_dim != 300:
                raise ValueError("Got embedding dim {}, expected size 300".format(embedding_dim))
            pretrained_type = vocab.FastText('en')

        embedding_len = len(self)
        weights = np.zeros((embedding_len, embedding_dim))
        words_found = 0

        for word, index in self.word2idx.items():
            try:
                # torchtext.vocab.__getitem__ defaults key error to a zero vector
                weights[index] = pretrained_type.vectors[pretrained_type.stoi[word]]
                words_found += 1
            except KeyError:
                if index == 0:
                    continue
                weights[index] = np.random.normal(scale=0.6, size=(embedding_dim))

        print(embedding_len - words_found, "words missing from pretrained")
        return torch.from_numpy(weights).float()


class RNNConfig:
    """ Helper class for model configuration."""

    def __init__(self, config_dict):
        # config_dict: a dict object holding configurations
        self.config = config_dict

    def __getattr__(self, item):
        return self.config[item]

    def update(self, new_config):
        self.config.update(new_config)
