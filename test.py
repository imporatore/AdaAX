import os

from config import DFA_DIR, IMAGE_DIR
from States import build_start_state
from DFA import DFA
from Pattern import PatternIterator


PATH = r'C:\PycharmProjects\AdaAX\AdaAX-main\synthetic\input.txt'

if __name__ == "__main__":
    from utils import RNNLoader
    from AdaAX import build_dfa
    from data.utils import save2pickle

    fname, model = 'synthetic_data_1', 'gru'
    loader = RNNLoader(fname, model)
    start_state = build_start_state()
    dfa = DFA(loader.alphabet, start_state, absorb=True)

    patterns = [expr.split(',')[1:-1] for expr in open(PATH, 'r')]
    # Line 79 ~ Line 83 in input.txt is not a positive pattern
    patterns_iter = PatternIterator(loader, patterns)

    dfa = build_dfa(loader, dfa, patterns_iter, tau=1., delta=0.)

    save2pickle(DFA_DIR, dfa, "{}_{}".format(fname, model))

    dfa.plot(os.path.join(IMAGE_DIR, "{}_{}".format(fname, model)))

    pass
