import copy
import argparse
import os
from collections import deque

import numpy as np
import tqdm

from config import START_PREFIX, K, THETA, TAU, DELTA, DFA_DIR, IMAGE_DIR
from States_prefixes import build_start_state, build_accept_state
from DFA_prefixes import DFA
from Pattern import pattern_extraction, PositivePatternTree
from utils import d, RNNLoader
from Helpers import check_consistency
from data.utils import save2pickle


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
    Q_new = deque()  # New pure sets to add

    # new state shouldn't be created for the last symbol in a pattern, since it is the accept state
    for i, s in enumerate(p[:-1]):
        if s in dfa.delta[q1].keys():
            q1 = dfa.delta[q1][s]
            if START_PREFIX + p[:i + 1] not in q1.prefixes:
                q1.prefixes.append(START_PREFIX + p[:i + 1])
            if q1 == dfa.F:
                return Q_new
        else:
            q1 = dfa.add_new_state(START_PREFIX + p[:i + 1], h[i], prev=q1)
            Q_new.append(q1)

    dfa.add_transit(q1, p[-1], dfa.F)  # add transition of the last symbol to accept state
    dfa.F.prefixes.append(START_PREFIX + p)
    return Q_new


def build_dfa(loader, dfa, patterns, merge_start, merge_accept, tau, delta):
    """ Build DFA using extracted patterns

    Args:
        patterns: list[list], list of patterns

    Params:
        TAU: threshold for neighbour distance
        DELTA: threshold for merging fidelity loss
    """

    dfa_fidelity = dfa.fidelity(loader)

    for i, res in enumerate(tqdm.tqdm(patterns)):
        p, h, _ = res  # (pattern, hidden, support)

        if i == 26:
            pass

        # list of new states created by pattern
        # if START_SYMBOL, first symbol in pattern is START_SYMBOL
        # A_t is modified as a private attribute of dfa so that it can be mapped while deepcopy,
        # or I would have to write explicit mapping dict.
        dfa.A_t = add_pattern(dfa, p[len(START_PREFIX):], h[len(START_PREFIX):])

        if not dfa.A_t:
            print("Pattern %d already accepted. Pass." % (i + 1))
            continue

        while dfa.A_t:

            assert all([st in dfa.Q for st in dfa.A_t])  # todo: test

            # try merge new states in A_t
            q_t = dfa.A_t.popleft()
            N_t = {s: d(q_t._h, s._h) for s in dfa.Q if s not in (q_t, dfa.F, dfa.q0)}  # neighbours of q_t

            neighbours = sorted(N_t.keys(), key=lambda x: N_t[x])
            if merge_start and q_t not in (dfa.q0, dfa.F):  # start and accepting state shouldn't be merged
                neighbours = [dfa.q0] + neighbours
            if merge_accept and q_t not in (dfa.q0, dfa.F):
                neighbours = [dfa.F] + neighbours

            for s in neighbours:
                if N_t.get(s, 0) >= tau:  # threshold (Euclidean distance of hidden values) for merging states
                    break
                try:
                    new_dfa, _ = merge_states(dfa, q_t, s)  # create the DFA after merging
                except RuntimeError:
                    continue

                new_dfa_fidelity = new_dfa.fidelity(loader)
                # accept merging if fidelity loss below threshold
                if dfa_fidelity - new_dfa_fidelity <= delta:
                    print("Merged: dfa fidelity %f; new dfa fidelity %f" % (dfa_fidelity, new_dfa_fidelity))
                    dfa, dfa_fidelity = new_dfa, new_dfa_fidelity
                    break

        print("Pattern %d, current fidelity: %f" % (i + 1, dfa_fidelity))

    return dfa


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

    if new_dfa.q0 == new_dfa.F:
        raise RuntimeError("Start state & accepting state merged. Quit merging.")

    # update to-merge list
    if state1 in dfa.A_t:
        new_dfa.A_t.remove(mapped_state1)
        # if state2 not in dfa.A_t and state2 != dfa.q0 and state2 != dfa.F:
        if state2 not in dfa.A_t:
            new_dfa.A_t.append(mapped_state2)

    # update prefixes
    mapped_state2.prefixes.extend(mapped_state1.prefixes)

    # update state list
    new_dfa.Q.remove(mapped_state1)
    mapping[state1] = mapped_state2

    # update entering (state2) transitions
    prefixes = mapped_state1.prefixes.copy()
    while prefixes:
        prefix = prefixes.pop()
        if prefix == START_PREFIX:
            continue

        # try:
        #     s, parent = prefix[-1], new_dfa.prefix2state(prefix[:-1])
        # except ValueError:  # todo: examine if this only occurs when prefix exceeds accepting state
        #     # in this case, no forward transitions should be added since accepting state already reached
        #     continue

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
    forward = new_dfa.delta.pop(mapped_state1, {})

    if mapped_state2 != new_dfa.F:  # no exiting transitions should be added if the merging state is accepting state

        for s in forward.keys():
            if forward[s] == mapped_state1:  # self-loop
                forward[s] = mapped_state2

        while forward:
            s, child = forward.popitem()
            if s not in new_dfa.delta[mapped_state2].keys():
                new_dfa.add_transit(mapped_state2, s, child)
            elif new_dfa.delta[mapped_state2][s] != child:
                new_dfa, mapping_ = merge_states(new_dfa, child, new_dfa.delta[mapped_state2][s], inplace=True)

                # update mapping
                mapping = {s: mapping_[ns] for s, ns in mapping.items()}

                # update states
                mapped_state2 = mapping_[mapped_state2]
                for s in forward.keys():
                    forward[s] = mapping_[forward[s]]

    if new_dfa.F in new_dfa.delta.keys():
        del new_dfa.delta[new_dfa.F]
        # also have to delete prefixes which are unreachable (beyond the accepting state)

    # if not inplace:  # Only check consistency when all merging is done
    #     check_consistency(new_dfa, check_transition=False, check_state=True, check_empty=True, check_null_states=True)
    # check_consistency(new_dfa, check_transition=True, check_state=True, check_empty=True, check_null_states=True)
    new_dfa._check_null_states()

    return new_dfa, mapping


def main(config):

    # create dfa & image directory
    if not os.path.exists(config.dfa_dir):
        os.makedirs(config.dfa_dir)
    if not os.path.exists(config.image_dir):
        os.makedirs(config.image_dir)

    loader = RNNLoader(config.fname, config.model)

    start_state = build_start_state()
    start_state._h = start_state.h(loader)
    accept_state = build_accept_state()
    accept_state._h = np.mean(loader.hidden_states[loader.rnn_output == 1, -1, :], axis=0)

    dfa = DFA(loader.alphabet, start_state, accept_state)

    patterns, support = pattern_extraction(
        loader, cluster_num=config.clusters, pruning=config.pruning, remove_padding=True)
    pattern_tree = PositivePatternTree(loader.prefix_tree)
    pattern_tree.update_patterns(patterns, support)

    dfa = build_dfa(loader, dfa, pattern_tree, config.merge_start, config.merge_accept,
                    config.neighbour, config.fidelity_loss)

    save2pickle(config.dfa_dir, dfa, "{}_{}_prefixes".format(config.fname, config.model))

    if config.plot:
        dfa.plot(os.path.join(config.image_path, "{}_{}_prefixes".format(config.fname, config.model)))

    return dfa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup parameters
    parser.add_argument("--fname", type=str, default="tomita_data_1")
    parser.add_argument("--model", type=str, default="rnn", choices=["rnn", "lstm", "gru", "glove-lstm"])
    parser.add_argument("--dfa_dir", type=str, default=DFA_DIR)
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    parser.add_argument("--plot", type=bool, default=True)
    # parser.add_argument("--start_symbol", type=str, default=START_SYMBOL)

    # pattern parameters
    parser.add_argument("--clusters", type=int, default=K)
    parser.add_argument("--pruning", type=float, default=THETA)

    # AdaAX parameters
    parser.add_argument("--merge_start", type=bool, default=True)
    parser.add_argument("--merge_accept", type=bool, default=False)
    parser.add_argument("--neighbour", type=float, default=TAU)
    parser.add_argument("--fidelity_loss", type=float, default=DELTA)

    args = parser.parse_args()

    print(args)

    main(args)
