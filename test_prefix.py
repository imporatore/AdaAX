import os

from config import DFA_DIR, IMAGE_DIR
from States_prefixes import build_start_state
from DFA_prefixes import DFA
from Pattern import PatternInputer


PATH = r'C:\PycharmProjects\AdaAX\AdaAX-main\synthetic\input.txt'

if __name__ == "__main__":
    from utils import RNNLoader
    from AdaAX_prefixes import build_dfa
    from data.utils import save2pickle

    fname, model = 'synthetic_data_1', 'gru'
    loader = RNNLoader(fname, model)
    start_state = build_start_state()
    dfa = DFA(loader.alphabet, start_state, absorb=True)

    patterns = [expr.split(',')[1:-1] for expr in open(PATH, 'r')]
    # Line 79 ~ Line 83 in input.txt is not a positive pattern
    patterns_iter = PatternInputer(loader, patterns)

    dfa = build_dfa(loader, dfa, patterns_iter, tau=1., delta=0.)

    save2pickle(DFA_DIR, dfa, "{}_{}_prefixes".format(fname, model))

    dfa.plot(os.path.join(IMAGE_DIR, "{}_{}_prefixes".format(fname, model)))

    pass
