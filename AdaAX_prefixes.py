import os
import copy
import argparse
import warnings
from collections import deque

import tqdm

from config import POS_THRESHOLD, SAMPLE_THRESHOLD, TAU, DELTA, DFA_DIR, IMAGE_DIR
from States_prefixes import build_start_state
from DFA_prefixes import DFA
from Pattern import PatternSampler
from utils import d, RNNLoader
from Helpers import substitute, check_consistency
from data.utils import save2pickle


# todo: if the merging stage is cooperated in the adding stage, would it be faster?
# todo: explicitly recalculate fidelity after adding a pattern using parsed result (mapping, missing)
def add_pattern(dfa, p, h):
    """ Add new pattern to DFA,

    return a new dfa added the pattern (so that abnormal pattern which cause magnificent fidelity loss can be rejected).

    Args:
        dfa: DFA
        p: list, pattern is a list of symbols
        h: list, list of hidden values of each prefix in p

    Returns:
        new_dfa: DFA, new dfa which added the pattern
        Q_new: list, new states (pure sets) added by the pattern
    """
    new_dfa = copy.deepcopy(dfa)
    q1 = new_dfa.q0
    Q_new = deque()  # New pure sets to add

    for i, s in enumerate(p):
        if s in new_dfa.delta[q1].keys():
            q1 = new_dfa.delta[q1][s]
            if p[:i + 1] not in q1.prefixes:
                q1.prefixes.append(p[:i + 1])
            # for absorb=True DFAs, no following transitions after accept states reached
            if new_dfa.absorb and q1 in new_dfa.F:
                return new_dfa, Q_new
        else:
            # add new states for missing transitions of a pattern
            q1 = new_dfa.add_new_state(p[:i + 1], h[i], prev=q1)
            Q_new.append(q1)

    if q1 not in new_dfa.F:
        if len(Q_new) == 0:  # all transitions found, but hasn't reached accept states
            warnings.warn("Try accept existing states for pattern %s." % p)
        new_dfa.F.append(q1)  # add new accept states

    return new_dfa, Q_new


def build_dfa(loader, dfa, patterns, tau, delta):
    """ Build DFA using extracted patterns.

    Args:
        loader: RNNLoader
        dfa: DFA
        patterns: Iterable, pattern iterator which yields Tuple(pattern, hidden, support)

    Params:
        tau: float, threshold for neighbour distance (Euclidean distance of hidden values)
        delta: float, threshold for merging fidelity loss
    """

    dfa_fidelity = dfa.fidelity(loader)

    for i, res in enumerate(tqdm.tqdm(patterns)):
        p, h, _ = res  # (pattern, hidden, support)

        new_dfa, states_to_be_merged = add_pattern(dfa, p, h)

        if not states_to_be_merged and len(new_dfa.F) == len(dfa.F):
            # print("Pattern %d already accepted. Pass." % (i + 1))
            continue
        else:
            new_dfa_fidelity = new_dfa.fidelity(loader)
            if new_dfa_fidelity >= dfa_fidelity:  # only accept patterns which increase fidelity
                # A_t is modified as an attribute of dfa so that it can be mapped while deepcopy
                dfa, dfa_fidelity, dfa.A_t = new_dfa, new_dfa_fidelity, states_to_be_merged
            else:
                print("Pattern %d: %s unaccepted." % (i, p))
                continue

        while dfa.A_t:

            q_t = dfa.A_t.popleft()  # pop elder states first for better performance
            if q_t != dfa.q0:
                N_t = {s: d(q_t.h, s.h) for s in dfa.Q if s not in [q_t, dfa.q0] + dfa.F}  # neighbours of q_t
            else:
                N_t = {s: 0 for s in dfa.Q if s not in [dfa.q0] + dfa.F}
            neighbours = sorted(N_t.keys(), key=lambda x: N_t[x])

            # when absorb=True, start and accept states shouldn't be merged
            if not dfa.absorb or q_t not in [dfa.q0] + dfa.F:
                neighbours = [state for state in dfa.F + [dfa.q0] if state != q_t] + neighbours
            elif q_t in dfa.F:
                neighbours = [state for state in dfa.F if state != q_t] + neighbours

            for s in neighbours:
                if N_t.get(s, 0) >= tau:  # threshold reached
                    break

                try:
                    new_dfa, _ = merge_states(dfa, q_t, s)  # create the DFA after merging
                except RuntimeError as message:  # Start state & accept states merged
                    if message.args[0] == "Start state & accept state merged for absorb=True DFA. Quit merging.":
                        continue
                    raise RuntimeError(message)

                new_dfa_fidelity = new_dfa.fidelity(loader)
                # accept merging if fidelity loss below threshold
                if dfa_fidelity - new_dfa_fidelity <= delta:
                    print("Merged: dfa fidelity %f; new dfa fidelity %f" % (dfa_fidelity, new_dfa_fidelity))
                    dfa, dfa_fidelity = new_dfa, new_dfa_fidelity
                    break

        print("Pattern %d, current fidelity: %f" % (i + 1, dfa_fidelity))

    check_consistency(dfa, check_transition=False, check_state=True, check_empty=True, check_null_states=True)
    print("Finished, extracted DFA fidelity: %f." % dfa.fidelity(loader))

    return dfa


def merge_states(dfa, state1, state2, inplace=False):
    """ Try merging state1 with state2 for given DFA.

    Notice that if the child states not consistent, they will also be merged.

    Args:
        dfa: DFA
        state1: State, the state to be merged
        state2: State, the state to merge
        inplace: bool, whether the modification would be in-place
            - should only be True when it is a child state merging

    Returns:
        new_dfa: DFA, new DFA after merging state1 with state2 in the existing DFA
        mapping: dict, the mapping from the states of dfa to the corresponding states of new dfa
    """
    # todo: the hidden state values remains after merging
    new_dfa = copy.deepcopy(dfa) if not inplace else dfa
    mapping = {s: ns for s, ns in zip(dfa.Q, new_dfa.Q)}
    mapped_state1, mapped_state2 = mapping[state1], mapping[state2]

    # Update start and accept states if merged.
    if state1 == dfa.q0:
        new_dfa.q0 = mapped_state2
    if state1 in dfa.F:
        substitute(new_dfa.F, mapped_state1, mapped_state2)

    # check for absorb=True DFA
    if new_dfa.absorb and new_dfa.q0 in new_dfa.F:
        raise RuntimeError("Start state & accept state merged for absorb=True DFA. Quit merging.")

    # update to-merge list
    substitute(new_dfa.A_t, mapped_state1, mapped_state2)

    # update prefixes
    mapped_state2.prefixes.extend(mapped_state1.prefixes)

    # update state list
    new_dfa.Q.remove(mapped_state1)
    mapping[state1] = mapped_state2

    # update entering (state2) transitions
    prefixes = mapped_state1.prefixes.copy()
    while prefixes:
        prefix = prefixes.pop()
        if not prefix:  # prefix [] for start state
            continue

        s, parent = prefix[-1], new_dfa.prefix2state(prefix[:-1])
        if parent == mapped_state1:
            # Note that this is a self loop and may encounter conflict for exiting transitions
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

    # no exiting transitions should be added if the merging state is an accept state when absorb=True
    if not new_dfa.absorb or mapped_state2 not in new_dfa.F:

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

                # update states & mapping
                mapped_state2 = mapping_[mapped_state2]
                for s in forward.keys():
                    forward[s] = mapping_[forward[s]]

    if not inplace:  # Only check consistency when all merging is done
        check_consistency(new_dfa, check_transition=False, check_state=True, check_empty=True, check_null_states=True)

    return new_dfa, mapping


def main(config):

    # create dfa & image directory
    if not os.path.exists(config.dfa_dir):
        os.makedirs(config.dfa_dir)
    if not os.path.exists(config.image_dir):
        os.makedirs(config.image_dir)

    loader = RNNLoader(config.fname, config.model)
    pattern_sampler = PatternSampler(loader, absorb=config.absorb, pos_threshold=config.pos_threshold,
                                     sample_threshold=config.sample_threshold, return_sample=config.add_single_sample)
    start_state = build_start_state()
    dfa = DFA(loader.alphabet, start_state, config.absorb)

    dfa = build_dfa(loader, dfa, pattern_sampler, config.neighbour, config.fidelity_loss)

    save2pickle(config.dfa_dir, dfa, "{}_{}_prefixes".format(config.fname, config.model))

    if config.plot:
        dfa.plot(os.path.join(config.image_dir, "{}_{}_prefixes".format(config.fname, config.model)))

    return dfa


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup parameters
    parser.add_argument("--fname", type=str)
    parser.add_argument("--model", type=str, choices=["rnn", "lstm", "gru", "glove-lstm"])
    parser.add_argument("--dfa_dir", type=str, default=DFA_DIR)
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR)
    parser.add_argument("--plot", type=bool, default=True)

    # pattern parameters
    parser.add_argument("--pos_threshold", type=float, default=POS_THRESHOLD)
    parser.add_argument("--sample_threshold", type=int, default=SAMPLE_THRESHOLD)
    parser.add_argument("--add_single_sample", type=bool, default=False)

    # AdaAX parameters
    parser.add_argument("--absorb", type=bool, default=True)
    parser.add_argument("--neighbour", type=float, default=TAU)
    parser.add_argument("--fidelity_loss", type=float, default=DELTA)

    args = parser.parse_args()

    print(args)

    main(args)
