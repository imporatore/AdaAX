import itertools
import random
import os

import pickle
import pandas as pd
import numpy as np


def gen_seq_fixed_len(alphabet, N_samples, length):
    """ Generate sequence for given alphabet, sample num and length.

    Args:
        - alphabet: str/list, symbols used when generating sequences
        - N_samples: int, number of samples generated
        - length: int, fix length for sequences generated

    Returns:
        - seq: list, generated input sequence of shape (N_sample, length)
    """
    if not isinstance(alphabet, str) and not isinstance(alphabet[0], str):
        alphabet = [str(s) for s in alphabet]

    # generate all possible permutations if N_samples exceeds
    if N_samples >= pow(len(alphabet), length):
        seq = [''.join(list(b)) for b in itertools.product(alphabet, repeat=length)]
    else:
        seq = []
        while len(seq) < N_samples:
            expr = ""
            for _ in range(length):
                expr += random.choice(alphabet)
            seq.append(expr)

    random.shuffle(seq)
    return seq


def gen_seq(alphabet, lens, num_per_len):
    """ Generate sequences for given alphabet and multiple lengths and sizes

    Note that the actual size of samples generated may less than len(lens) * num_per_len,
    since that for smaller length the possible permutation num may be inadequate.

    Args:
        - alphabet: list, symbols used when generating sequences
        - lens: list, lengths for sequences generated
        - num_per_len: int, sample num per length

    Returns:
        - seq: list, generated input sequence of shape (N_sample, length)
    """
    seq = []
    for l in lens:
        seq += gen_seq_fixed_len(alphabet, num_per_len, l)

    random.shuffle(seq)
    return seq


def gen_dataset(alphabet, rule, lens, num_per_len, class_balance=True):
    """ Generate a (class-balanced) training set for given alphabet and multiple lengths and sizes.

    Labels are evaluated using 'rule' function.
    Note that the actual size of samples generated may less than len(lens) * num_per_len,
    due to inadequate possible permutations and class balance procedure.

    Args:
        - alphabet: list, symbols used when generating sequences
        - rule: func, target is positive (1) iff rule(expr)
        - lens: list, lengths for sequences generated
        - num_per_len: int, sample num per length
        - class_balance: bool, if dataset generated should be class balanced

    Returns:
        - X: list, generated class-balanced input sequence
        - y: list, labels evaluated using func 'rule'
    """

    if isinstance(lens, int):
        lens = [lens]

    X, y = [], []
    for l in lens:
        seq = gen_seq_fixed_len(alphabet, num_per_len, l)
        pos_idx = [i for i in range(len(seq)) if rule(seq[i])]
        neg_idx = [i for i in range(len(seq)) if i not in pos_idx]

        if class_balance:
            pos_idx = random.sample(pos_idx, min(len(pos_idx), len(neg_idx)))
            neg_idx = random.sample(neg_idx, min(len(pos_idx), len(neg_idx)))

        X += [seq[i] for i in pos_idx]
        y += [1] * len(pos_idx)
        X += [seq[i] for i in neg_idx]
        y += [0] * len(neg_idx)

    print("Made train set of size: %d." % len(y))
    rand_order = random.sample(range(len(X)), len(X))
    return [X[i] for i in rand_order], [y[i] for i in rand_order]


def save2pickle(fpath, data, fname):

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    with open(os.path.join(fpath, fname + ".pickle"), "wb") as handle:
        pickle.dump(data, handle)
        # pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    print("Successfully dumped %s in %s." % (fname, fpath))


def load_pickle(fpath, fname):
    with open(os.path.join(fpath, fname + ".pickle"), 'rb') as handle:
        data = pickle.load(handle)
    handle.close()
    return data


def save2csv(fpath, data, fname):

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    if isinstance(data, tuple):
        data = pd.DataFrame(data, columns=['expr', 'label'])
    data.to_csv(os.path.join(fpath, fname + ".csv"), index=False)


def load_csv(fpath, fname):
    data = pd.read_csv(os.path.join(fpath, fname + ".csv"))
    return data


def save2npy(fpath, data, fname):

    if not os.path.exists(fpath):
        os.makedirs(fpath)

    np.save(os.path.join(fpath, fname + '.npy'), data)
