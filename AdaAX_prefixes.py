import copy

import numpy as np
import tqdm

from config import START_PREFIX, TAU, DELTA
from States_prefixes import build_start_state, build_accept_state
from DFA_prefixes import DFA
from Pattern import pattern_extraction, PositivePatternTree
from utils import d
from Helpers import check_consistency


# todo: if the merging stage is cooperated in the adding stage, would it be faster?
def add_pattern(dfa, p, h):
    """ Add new pattern to DFA

    Args:
        dfa: DFA, the dfa modified in-place
        p: list, pattern is a list of symbols
        h: list, list of hidden values of each prefix in p

    Returns:
        Q_new: list, new states (pure sets) added by the pattern
    """
    q1 = dfa.q0
    Q_new = []  # New pure sets to add

    # new state shouldn't be created for the last symbol in a pattern, since it is the accept state
    for i, s in enumerate(p[:-1]):
        if s in dfa.delta[q1].keys():
            q1 = dfa.delta[q1][s]
        else:
            q1 = dfa.add_new_state(START_PREFIX + p[:i + 1], h[i], prev=q1)
            Q_new.append(q1)

    dfa.add_transit(q1, p[-1], dfa.F)  # add transition of the last symbol to accept state
    dfa.F.prefixes.append(START_PREFIX + p)
    return Q_new


def build_dfa(loader, dfa, patterns, merge_start, merge_accept):
    """ Build DFA using extracted patterns

    Args:
        patterns: list[list], list of patterns

    Params:
        TAU: threshold for neighbour distance
        DELTA: threshold for merging fidelity loss
    """
    for p, h, _ in tqdm.tqdm(patterns):  # (pattern, hidden, support)
        # list of new states created by pattern
        # if START_SYMBOL, first symbol in pattern is START_SYMBOL
        # A_t is modified as a private attribute of dfa so that it can be mapped while deepcopy,
        # or I would have to write explicit mapping dict.
        dfa.A_t = add_pattern(dfa, p[len(START_PREFIX):], h[len(START_PREFIX):])
        while dfa.A_t:

            assert all([st in dfa.Q for st in dfa.A_t])  # todo: test

            # try merge new states in A_t
            q_t = dfa.A_t.pop()
            if merge_start:
                if merge_accept:
                    N_t = {s: d(q_t._h, s._h) for s in dfa.Q if s != q_t}  # neighbours of q_t
                else:
                    N_t = {s: d(q_t._h, s._h) for s in dfa.Q if s not in (q_t, dfa.F)}
            else:
                if merge_accept:
                    N_t = {s: d(q_t._h, s._h) for s in dfa.Q if s not in (q_t, dfa.q0)}
                else:
                    N_t = {s: d(q_t._h, s._h) for s in dfa.Q if s not in (q_t, dfa.F, dfa.q0)}

            for s in sorted(N_t.keys(), key=lambda x: N_t[x]):
                if N_t[s] >= TAU:  # threshold (Euclidean distance of hidden values) for merging states
                    break

                new_dfa, _ = merge_states(dfa, q_t, s)  # create the DFA after merging
                # accept merging if fidelity loss below threshold
                if dfa.fidelity(loader) - new_dfa.fidelity(loader) < DELTA:
                    dfa = new_dfa
                    break


# todo: require testing
# todo: test for self-loop and transition to the state it merges with
def merge_states(dfa, state1, state2, inplace=False):
    """ Try merging state1 with state2.

    Notice that if the child state not consistent, they will also be merged.

    Args:
        state1:
        state2:

    Returns:
        new_dfa: new DFA after merging state1 with state2 in the existing DFA
    """
    # todo: forbid merging accept state
    # todo: add threshold for merging accept state
    # todo: the hidden state values remains after merging
    new_dfa = copy.deepcopy(dfa) if not inplace else dfa
    mapping = {s: ns for s, ns in zip(dfa.Q, new_dfa.Q)}
    mapped_state1, mapped_state2 = mapping[state1], mapping[state2]

    # todo: check if start & accept state would be merged
    # Update start and accept states if merged.
    if state1 == dfa.q0:
        new_dfa.q0 = mapped_state2
    elif state1 == dfa.F:
        new_dfa.F = mapped_state2

    # update to-merge list
    if state1 in dfa.A_t:
        new_dfa.A_t.remove(mapped_state1)
        if state2 not in dfa.A_t:
            new_dfa.A_t.append(mapped_state2)

    # update prefixes
    mapped_state2.prefixes.extend(mapped_state1.prefixes)

    # update state list
    new_dfa.Q.remove(mapped_state1)
    mapping[state1] = mapped_state2

    # update entering (state2) transitions
    prefixes = mapped_state1.prefixes
    while prefixes:
        prefix = prefixes.pop()
        s, parent = prefix[-1], new_dfa.prefix2state(prefix[:-1])
        if parent == mapped_state1:  # todo: note that this is a self loop and may encounter conflict for exiting transitions
            pass  # Note that this self-loop will also exist when updating exiting transitions
        else:  # since the transition is deterministic, they MUST NOT be state2's parents
            new_dfa.add_transit(parent, s, mapped_state2)
        for p in parent.prefixes:
            try:
                prefixes.remove(p + [s])
            except ValueError:
                pass

    # update exiting transitions
    forward = new_dfa.delta.pop(mapped_state1)
    for s, c in forward.items():
        child = mapped_state2 if c == mapped_state1 else c  # self-loop
        if s not in new_dfa.delta[mapped_state2].keys():
            new_dfa.add_transit(mapped_state2, s, child)
        elif new_dfa.delta[mapped_state2][s] != child:
            new_dfa = merge_states(new_dfa, child, new_dfa.delta[mapped_state2][s], inplace=True)

    check_consistency(new_dfa, check_transition=False, check_state=True, check_empty=True)

    return new_dfa


def main(rnn_loader, merge_start=True, merge_accept=True, plot=True):

    start_state = build_start_state()
    start_state._h = start_state.h(rnn_loader)
    accept_state = build_accept_state()
    accept_state._h = np.mean(rnn_loader.hidden_states[rnn_loader.rnn_output == 1, -1, :], axis=0)

    dfa = DFA(rnn_loader.alphabet, start_state, accept_state)

    patterns, support = pattern_extraction(rnn_loader, remove_padding=True)
    pattern_tree = PositivePatternTree(rnn_loader.prefix_tree)
    pattern_tree.update_patterns(patterns, support)

    build_dfa(rnn_loader, dfa, pattern_tree, merge_start, merge_accept)


    if plot:
        dfa.plot()

    return dfa


if __name__ == "__main__":
    from utils import RNNLoader

    loader = RNNLoader('tomita_data_1', 'gru')
    main(loader)
