import os
import copy
import argparse
import warnings
from collections import deque, defaultdict
from typing import Tuple

import tqdm

from config import POS_THRESHOLD, SAMPLE_THRESHOLD, TAU, DELTA, DFA_DIR, IMAGE_DIR
from States import build_start_state
from DFA import DFA
from Pattern import PatternSampler
from utils import RNNLoader
from Helpers import d, substitute, check_consistency
from data.utils import save2pickle


# todo: if the merging stage is cooperated in the adding stage, would it be faster? (Yes, as it may use existing links)
# todo: (No, as I have already adopted a bottom-up merging order)
def add_pattern(dfa: DFA, n, p, h):
    """ Add new pattern to DFA,

    return a new dfa added the pattern (so that abnormal pattern which cause magnificent fidelity loss can be rejected).

    Args:
        dfa: DFA
        n: list[Node], list of nodes
        p: list, pattern is a list of symbols
        h: list, list of hidden values of each prefix in p

    Returns:
        new_dfa: DFA, new dfa which added the pattern
        Q_new: list, new states (pure sets) added by the pattern
    """
    new_dfa = copy.copy(dfa)
    q1 = new_dfa.q0
    Q_new = deque()  # New pure sets to add

    for i, s in enumerate(p):
        if s in new_dfa.delta[q1].keys():  # exist transitions
            q1 = new_dfa.delta[q1][s]
            # for absorb=True DFAs, no following transitions after accept states reached
            if q1 in new_dfa.F and new_dfa.absorb:
                return new_dfa, Q_new, None
        else:
            # add new states for missing transitions of a pattern
            q1 = new_dfa.add_new_state(p[:i + 1], h[i], n[i].pos_sup, prev=q1)
            new_dfa.mapping.update({q1: {n[i]}})
            # new_dfa.missing.remove(n[i])
            # new_dfa.missing.update(n[i].next)
            Q_new.append(q1)

    if q1 not in new_dfa.F:  # last node unaccepted
        if len(Q_new) == 0:  # all transitions found, but hasn't reached accept states
            warnings.warn("Try accept existing states for pattern %s." % p)
        new_dfa.F.add(q1)  # add new accept states=
        return new_dfa, Q_new, q1

    return new_dfa, Q_new, None


def build_dfa(loader: RNNLoader, dfa: DFA, patterns, tau, delta, class_balanced):
    """ Build DFA using extracted patterns. Supports incremental update.

    Args:
        loader: RNNLoader
        dfa: DFA
        patterns: Iterable, pattern iterator which yields Tuple(pattern, hidden, support)
        class_balanced: bool, default=False, whether to calculate fidelity using class balanced weights

    Params:
        tau: float, threshold for neighbour distance (Euclidean distance of hidden values)
        delta: float, threshold for merging fidelity loss

    Note:
        Mapping & missing is tracked and updated explicitly during both the add pattern and consolidation process.
    """

    if len(dfa.delta) == 0:  # initialize mapping & missing
        dfa.mapping.update({dfa.q0: {loader.prefix_tree.root}})
        # dfa.missing.update(loader.prefix_tree.root.next)
    else:  # incremental update a dfa
        # dfa.mapping, dfa.missing = dfa.parse_tree(loader.prefix_tree.root, dfa.q0)
        dfa.mapping = defaultdict(set, dfa.parse_tree(loader.prefix_tree.root, dfa.q0))
    dfa.fidelity = dfa.eval_fidelity(loader, class_balanced)  # initialize fidelity

    for i, res in enumerate(tqdm.tqdm(patterns)):
        nodes, p, h, _ = res  # (nodes, pattern, hidden, support)

        new_dfa, states_to_be_merged, new_accept = add_pattern(dfa, nodes, p, h)

        if not states_to_be_merged and not new_accept:
            # print("Pattern %d already accepted. Pass." % (i + 1))
            continue
        else:
            for node in new_dfa.mapping[new_accept]:
                new_dfa.update_node_fidelity(loader, node, class_balanced)
            if new_dfa.fidelity >= dfa.fidelity:  # only accept patterns which increase fidelity
                # A_t is modified as an attribute of dfa so that it can be mapped while deepcopy
                dfa, dfa.A_t = new_dfa, states_to_be_merged
            else:
                print("Pattern %d: %s unaccepted." % (i, p))
                continue

        while dfa.A_t:

            q_t = dfa.A_t.popleft()  # pop elder states first for better performance
            if q_t != dfa.q0:
                N_t = {s: d(q_t.h, s.h) for s in dfa.Q if s not in dfa.F.union({q_t, dfa.q0})}  # neighbours of q_t
            else:
                N_t = {s: 0 for s in dfa.Q if s not in dfa.F.union({dfa.q0})}
            neighbours = sorted(N_t.keys(), key=lambda x: N_t[x])

            # when absorb=True, start and accept states shouldn't be merged
            if not dfa.absorb or q_t not in dfa.F.union({dfa.q0}):
                neighbours = [state for state in dfa.F.union({dfa.q0}) if state != q_t] + neighbours
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
                    elif message.args[0] == "No accept state remains.":
                        continue
                    raise RuntimeError(message)

                new_dfa.fidelity = new_dfa.eval_fidelity(loader, class_balanced)
                # accept merging if fidelity loss below threshold
                if dfa.fidelity - new_dfa.fidelity <= delta:
                    print("Pattern %d Merged: dfa fidelity %f; new dfa fidelity %f" %
                          (i + 1, dfa.fidelity, new_dfa.fidelity))
                    dfa = new_dfa
                    break

        print("Pattern %d, current fidelity: %f" % (i + 1, dfa.fidelity))

    check_consistency(dfa, check_transition=True, check_state=True, check_empty=True, check_null_states=True)
    print("Finished, extracted DFA fidelity: %f." % dfa.eval_fidelity(loader, class_balanced))

    return dfa


def merge_states(dfa: DFA, state1, state2, inplace=False) -> Tuple[DFA, dict]:
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

    Notes:
        Four cases:
        1. State 1 has outgoing symbol 's' while state 2 does not
        2. State 2 has outgoing symbol 's' while state 1 does not
        3. State 1 has common outgoing symbol 's' with state 2, and child state the same
        4. State 1 has common outgoing symbol 's' with state 2, and child state not the same
    """
    # todo: the hidden state values remains after merging
    new_dfa = copy.copy(dfa) if not inplace else dfa
    mapping = {s: ns for s, ns in zip(dfa.Q, new_dfa.Q)}
    mapped_state1, mapped_state2 = mapping[state1], mapping[state2]
    weight1, weight2 = mapped_state1.weight, mapped_state2.weight
    mapped_state2.h = (weight1 * mapped_state1.h + weight2 * mapped_state2.h) / (weight1 + weight2)
    mapped_state2.weight = weight1 + weight2

    # update accept mapping for absorb=True DFA
    if new_dfa.absorb:
        if state1 in dfa.F and state2 not in dfa.F:
            for node in new_dfa.mapping[mapped_state2].copy():
                new_dfa.remove_descendants(node, mapped_state2)
        if state2 in dfa.F and state1 not in dfa.F:
            for node in new_dfa.mapping[mapped_state1].copy():
                new_dfa.remove_descendants(node, mapped_state1)

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

    # update state list
    new_dfa.Q.remove(mapped_state1)
    mapping[state1] = mapped_state2

    # update state-nodes mapping
    state_1_nodes = new_dfa.mapping.pop(mapped_state1)
    new_dfa.mapping[mapped_state2].update(state_1_nodes)

    # update transition table
    forward, backward = new_dfa.delta.pop(mapped_state1)

    # update entering (state2) transitions`
    for s in backward.keys():
        for parent in backward[s]:
            if parent == mapped_state1:
                # Note that this is a self loop and may encounter conflict for exiting transitions
                pass  # Note that this self-loop will also exist when updating exiting transitions
            else:  # since the transition is deterministic, they MUST NOT be state2's parents
                new_dfa.add_transit(parent, s, mapped_state2)

    # update exiting transitions
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

    # update descendant mapping
    if not new_dfa.absorb or mapped_state2 not in new_dfa.F:

        for node in new_dfa.mapping[mapped_state2].copy():
            for n in node.next:
                # parse unfounded nodes with newly built transitions
                if n.val in new_dfa.delta[mapped_state2].keys():
                    state = new_dfa.delta[mapped_state2][n.val]
                    if n not in new_dfa.mapping[state]:
                        # new_dfa.missing.remove(n)
                        # mapping, missing = new_dfa.parse_tree(n, state)
                        # new_dfa.update_mapping(mapping, missing)
                        state2nodes = new_dfa.parse_tree(n, state)
                        new_dfa.update_mapping(state2nodes)

    if not inplace:  # Only check consistency when all merging is done
        check_consistency(new_dfa, check_transition=True, check_state=True, check_empty=True, check_null_states=True)

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
    start_state = build_start_state(loader)
    dfa = DFA(loader.alphabet, start_state, config.absorb)

    dfa = build_dfa(loader, dfa, pattern_sampler, config.neighbour, config.fidelity_loss, config.class_balanced)

    save2pickle(config.dfa_dir, dfa, "{}_{}".format(config.fname, config.model))

    if config.plot:
        dfa.plot(os.path.join(config.image_dir, "{}_{}".format(config.fname, config.model)))

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
    parser.add_argument("--class_balanced", type=bool, default=False)
    parser.add_argument("--neighbour", type=float, default=TAU)
    parser.add_argument("--fidelity_loss", type=float, default=DELTA)

    args = parser.parse_args()

    print(args)

    main(args)
