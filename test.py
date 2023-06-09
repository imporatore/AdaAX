import os

import numpy as np

from config import START_PREFIX, TAU, DELTA, DFA_DIR, IMAGE_DIR
# from States import build_start_state, build_accept_state
from States_prefixes import build_start_state, build_accept_state
# from DFA import DFA
from DFA_prefixes import DFA
from Pattern import PatternIterator


PATH = r'C:\PycharmProjects\AdaAX\AdaAX-main\synthetic\input.txt'

if __name__ == "__main__":
    from utils import RNNLoader
    # from AdaAX import build_dfa
    from AdaAX_prefixes import build_dfa
    from data.utils import save2pickle

    fname, model = 'synthetic_data_1', 'gru'
    loader = RNNLoader(fname, model)
    start_state = build_start_state()
    start_state._h = start_state.h(loader)
    accept_state = build_accept_state()
    accept_state._h = np.mean(loader.hidden_states[loader.rnn_output == 1, -1, :], axis=0)

    dfa = DFA(loader.alphabet, start_state, accept_state)

    patterns = [START_PREFIX + expr.split(',')[1:-1] for expr in open(PATH, 'r')]
    patterns_iter = PatternIterator(loader.prefix_tree, patterns)

    build_dfa(loader, dfa, patterns_iter, merge_start=True, merge_accept=True, tau=TAU, delta=DELTA)

    # save2pickle(DFA_DIR, dfa, "{}_{}".format(fname, model))
    save2pickle(DFA_DIR, dfa, "{}_{}_prefixes".format(fname, model))

    # dfa.plot(os.path.join(IMAGE_DIR, "{}_{}".format(fname, model)))
    dfa.plot(os.path.join(IMAGE_DIR, "{}_{}_prefixes".format(fname, model)))

    pass