import copy

import numpy as np

from config import START_PREFIX, TAU, DELTA
from States import build_start_state, build_accept_state
from DFA import DFA
from Pattern import pattern_extraction
from utils import d


# todo: add pattern by PatternTree
# todo: require testing
# todo: if the merging stage is cooperated in the adding stage, would it be faster?
def add_pattern(dfa, p):
    """ Add new? pattern to DFA

    Args:
        p: list, pattern is a list of symbols

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
            q1 = dfa.add_new_state(START_PREFIX + p[:i + 1], q1)
            Q_new.append(q1)

    dfa.add_transit(q1, p[-1], dfa.F)  # add transition of the last symbol to accept state
    return Q_new
    # todo: add pattern by PatternTree
    # todo: require testing


def build_dfa(dfa, patterns, merge_start, merge_accept):
    """ Build DFA using extracted patterns

    Args:
        patterns: list[list], list of patterns

    Params:
        TAU: threshold for neighbour distance
        DELTA: threshold for merging fidelity loss
    """
    for p in patterns:
        # list of new states created by pattern
        A_t = dfa.add_pattern(p[len(START_PREFIX):])  # if START_SYMBOL, first symbol in pattern is START_SYMBOL
        while A_t:

            assert all([st in dfa.Q for st in A_t])

            # try merge new states in A_t
            q_t = A_t.pop()
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

                new_dfa = merge_states(dfa, q_t, s)  # create the DFA after merging
                if dfa.fidelity - new_dfa.fidelity < DELTA:  # accept merging if fidelity loss below threshold
                    dfa.Q, dfa.q0, dfa.F = new_dfa.Q, new_dfa.q0, new_dfa.F  # update states
                    dfa.delta = new_dfa.delta  # update transitions
                    A_t = [new_dfa.__mapping[state] for state in A_t]
                    break


# todo: require testing
# todo: test for self-loop and transition to the state it merges with
def merge_states(dfa, state1, state2):
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
    new_dfa = copy.deepcopy(dfa)
    mapping = {s: ns for s, ns in zip(dfa.Q, new_dfa.Q)}
    mapped_state1, mapped_state2 = mapping[state1], mapping[state2]
    new_state = copy.deepcopy(mapped_state2)

    for s in new_state.parents.keys():
        if mapped_state2 in new_state.parents[s]:
            new_state.parents[s].remove(mapped_state2)
            new_state.parents[s].append(new_state)
        if mapped_state1 in new_state.parents[s]:
            new_state.parents[s].remove(mapped_state1)
            if new_state not in new_state.parents[s]:
                new_state.parents[s].append(new_state)

    # todo: check if start & accept state would be merged
    # Update start and accept states if merged.
    if state2 == dfa.q0:
        new_dfa.q0 = new_state
    elif state2 == dfa.F:
        new_dfa.F = new_state

    # update children set
    for s in mapped_state1.parents.keys():
        for p in mapped_state1.parents[s]:
            if p not in new_state.parents[s]:
                if p == mapped_state1 or p == mapped_state2:  # self-loop
                    new_state.parents[s] += [new_state]
                else:
                    new_state.parents[s] += [p]

    # update states
    new_dfa.Q.append(new_state)
    new_dfa.Q.remove(mapped_state1)
    new_dfa.Q.remove(mapped_state2)

    # Update income transitions
    for s in new_state.parents.keys():
        for state in new_state.parents[s]:
            new_dfa.delta[state][s] = new_state
    # todo: seems no self loop with mapped_state1 & mapped_state2 exists
    # Update outgoing transitions
    transition1 = new_dfa.delta.pop(mapped_state1)
    transition2 = new_dfa.delta.pop(mapped_state2)
    for s in transition1.keys():
        child1 = transition1[s]
        if child1 == mapped_state1 or child1 == mapped_state2:
            child1 = new_state  #
        try:
            child2 = transition2.pop(s)
            if child2 == mapped_state2 or child2 == mapped_state1:
                child2 = new_state
            # todo: self loop
            if child1 != child2:
                # Merge outgoing states for common outgoing symbol if child state doesn't correspond
                new_dfa = new_dfa._merge_states(child1, child2)
            else:
                # update consistent child state
                new_dfa._add_transit(new_state, s, child1)
                if child1 != new_state:
                    child1.parents[s].remove(mapped_state1)
                    child2.parents[s].remove(mapped_state2)
        except KeyError:  # outgoing symbol only in state1
            if child1 == mapped_state1 or child1 == mapped_state2:
                new_dfa._add_transit(new_state, s, new_state)
            else:
                new_dfa._add_transit(new_state, s, child1)
                child1.parents[s].remove(mapped_state1)

    for s, child in transition2.items():  # outgoing symbol only in state2
        if child == mapped_state2 or child == mapped_state1:
            new_dfa._add_transit(new_state, s, new_state)
        else:
            new_dfa._add_transit(new_state, s, child)
            child.parents[s].remove(mapped_state2)

    new_dfa.__mapping[state1] = new_state
    new_dfa.__mapping[state2] = new_state

    for st in new_dfa.delta.keys():
        for s in new_dfa.delta.keys():
            assert all([c in new_dfa.Q for c in new_dfa.delta[st][s]])

    return new_dfa


def main(rnn_loader, merge_start=True, merge_accept=False):

    start_state = build_start_state()
    start_state._h = start_state.h(rnn_loader)
    accept_state = build_accept_state()
    accept_state._h = np.mean(rnn_loader.hidden_states[rnn_loader.rnn_output == 1, -1, :], axis=0)

    dfa = DFA(rnn_loader.alphabet, start_state, accept_state)

    def _build_add_state(prefixes, prev):
        state = dfa.add_new_state(prefixes, prev)
        state._h = state.h(rnn_loader)

    dfa.add_new_state = _build_add_state
    dfa.add_pattern = lambda p: add_pattern(dfa, p)

    patterns, support = pattern_extraction(rnn_loader, remove_padding=True)

    build_dfa(dfa, patterns, merge_start, merge_accept)

