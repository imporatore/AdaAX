import functools
import string
from collections import defaultdict
import copy

import graphviz as gv
from IPython.display import Image
from IPython.display import display

from states import State, build_start_state, build_accept_state
from utils import d
from config import TAU, DELTA


digraph = functools.partial(gv.Digraph, format='png')
graph = functools.partial(gv.Graph, format='png')

separator = "_"


class DFA:
    def __init__(self, rnn_loader):
        self.alphabet = rnn_loader.alphabet
        self.q0 = build_start_state()
        self.F = build_accept_state(rnn_loader)
        self.Q = [self.q0, self.F]
        self.delta = defaultdict(defaultdict)  # todo: require testing

        self._eval_fidelity = rnn_loader.eval_fidelity
        self._prefix2state = defaultdict()

    # todo: add pattern by PatternTree
    # todo: require testing
    def _add_pattern(self, p):
        """ Add new? pattern to """
        q1 = self.q0
        Q_new = []  # New pure sets to add
        for i, s in enumerate(p):
            if s in self.delta[q1].keys():
                q1 = self.delta[q1][s]
            else:
                q2 = State(p[:i+1])  # Initialize pure set from new prefix

                # Update DFA
                self.Q.append(q2)  # Add to states
                self.delta[q1][s] = q2  # Update transition
                self._prefix2state[p[:i+1]] = self._prefix2state[p[:i+1]] + [q2]  # Update prefix2state dict

                Q_new.append(q2)
                q1 = q2

        self.delta[q1][s] = self.F
        return Q_new

    # todo: add pattern by PatternTree
    # todo: require testing
    def build_dfa(self, patterns):
        for p in patterns:
            A_t = self._add_pattern(p)
            while A_t:
                q_t = A_t.pop()
                N_t = {s: d(q_t._h, s._h) for s in self.Q}
                for s in sorted(N_t.keys(), key=lambda x: N_t[x]):
                    if N_t[s] >= TAU:
                        break

                    new_dfa = self._merge_states(q_t, s)
                    if self.fidelity - new_dfa.fidelity < DELTA:
                        self.Q, self.q0, self.F, self.delta = new_dfa.Q, new_dfa.q0, new_dfa.F, new_dfa.delta
                        break

    # todo: require testing
    def _merge_states(self, state1, state2):
        # todo: forbid merging accept state
        # todo: add threshold for merging accept state
        # todo: the hidden state values remains after merging
        new_dfa, new_state = copy.copy(self), copy.copy(state2)
        # new_state._added_prefix = new_state._added_prefix + state1._prefix
        # new_state._added_prefix.extend(state1._added_prefix)
        new_state._prefix = state2.prefixes + state1.prefixes

        # update states
        new_dfa.Q.remove(state1)
        new_dfa.Q.remove(state2)
        new_dfa.Q.append(new_state)

        # Update prefix2state dict and income transitions
        for prefix in new_state._prefix:
            new_dfa._prefix2state[prefix] = new_state

            prev_prefix, s = prefix[:-1], prefix[-1]
            new_dfa.delta[new_dfa._prefix2state[prev_prefix]][s] = new_state

        s1, s2 = new_dfa.delta[state1].keys(), new_dfa.delta[state2].keys()
        # Merge outgoing states for common outgoing symbol
        for s in set(s1).intersection(set(s2)):
            new_dfa = new_dfa._merge_states(new_dfa.delta[state1][s], new_dfa.delta[state2][s])

        # Update outgoing transitions
        transition1, transition2 = new_dfa.delta.pop(state1), new_dfa.delta.pop(state2)
        for s in transition1.keys():
            new_dfa.delta[new_state][s] = transition1[s]
        for s in transition2.keys():
            new_dfa.delta[new_state][s] = transition2[s]

        # Update start and accept states if merged.
        if state2 == new_dfa.q0:
            new_dfa.q0 = new_state
        if state2 == new_dfa.F:
            new_dfa.F = new_state

        return new_dfa

    # todo: require testing
    def classify_expression(self, expression):
        # Expression is string with only letters in alphabet
        q = self.q0
        for s in expression:
            q = self.delta[q][s]
        return q == self.F

    # todo: require testing
    @property
    def fidelity(self):
        return self._eval_fidelity(self)

    # todo: require modify & testing. F used to be a set of accept states...
    def draw_nicely(self, force=False, maximum=60):
        # Stolen from Lstar
        # todo: if two edges are identical except for letter, merge them and note both the letters
        if (not force) and len(self.Q) > maximum:
            return

        # suspicion: graphviz may be upset by certain sequences, avoid them in nodes
        label_to_number_dict = {False: 0}  # false is never a label but gets us started

        def label_to_numberlabel(label):
            max_number = max(label_to_number_dict[l] for l in label_to_number_dict)
            if not label in label_to_number_dict:
                label_to_number_dict[label] = max_number + 1
            return str(label_to_number_dict[label])

        def add_nodes(graph, nodes):  # stolen from http://matthiaseisen.com/articles/graphviz/
            for n in nodes:
                if isinstance(n, tuple):
                    graph.node(n[0], **n[1])
                else:
                    graph.node(n)
            return graph

        def add_edges(graph, edges):  # stolen from http://matthiaseisen.com/articles/graphviz/
            for e in edges:
                if isinstance(e[0], tuple):
                    graph.edge(*e[0], **e[1])
                else:
                    graph.edge(*e)
            return graph

        g = digraph()
        g = add_nodes(g, [(label_to_numberlabel(self.q0), {'color': 'green' if self.q0 in self.F else 'black',
                                                           'shape': 'hexagon', 'label': 'start'})])
        states = list(set(self.Q) - {self.q0})
        g = add_nodes(g, [(label_to_numberlabel(state), {'color': 'green' if state in self.F else 'black',
                                                         'label': str(i)})
                          for state, i in zip(states, range(1, len(states) + 1))])

        def group_edges():
            def clean_line(line, group):
                line = line.split(separator)
                line = sorted(line) + ["END"]
                in_sequence = False
                last_a = ""
                clean = line[0]
                if line[0] in group:
                    in_sequence = True
                    first_a = line[0]
                    last_a = line[0]
                for a in line[1:]:
                    if in_sequence:
                        if a in group and (ord(a) - ord(last_a)) == 1:  # continue sequence
                            last_a = a
                        else:  # break sequence
                            # finish sequence that was
                            if (ord(last_a) - ord(first_a)) > 1:
                                clean += ("-" + last_a)
                            elif not last_a == first_a:
                                clean += (separator + last_a)
                            # else: last_a==first_a -- nothing to add
                            in_sequence = False
                            # check if there is a new one
                            if a in group:
                                first_a = a
                                last_a = a
                                in_sequence = True
                            if not a == "END":
                                clean += (separator + a)
                    else:
                        if a in group:  # start sequence
                            first_a = a
                            last_a = a
                            in_sequence = True
                        if not a == "END":
                            clean += (separator + a)
                return clean

            edges_dict = {}
            for state in self.Q:
                for a in self.alphabet:
                    edge_tuple = (label_to_numberlabel(state), label_to_numberlabel(self.delta[state][a]))
                    # print(str(edge_tuple)+"    "+a)
                    if not edge_tuple in edges_dict:
                        edges_dict[edge_tuple] = a
                    else:
                        edges_dict[edge_tuple] += separator + a
                    # print(str(edge_tuple)+"  =   "+str(edges_dict[edge_tuple]))
            for et in edges_dict:
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_lowercase)
                edges_dict[et] = clean_line(edges_dict[et], string.ascii_uppercase)
                edges_dict[et] = clean_line(edges_dict[et], "0123456789")
                edges_dict[et] = edges_dict[et].replace(separator, ",")
            return edges_dict

        edges_dict = group_edges()
        g = add_edges(g, [(e, {'label': edges_dict[e]}) for e in edges_dict])
        # print('\n'.join([str(((str(state),str(self.delta[state][a])),{'label':a})) for a in self.alphabet for state in
        #                  self.Q]))
        # g = add_edges(g,[((label_to_numberlabel(state),label_to_numberlabel(self.delta[state][a])),{'label':a})
        #                  for a in self.alphabet for state in self.Q])
        display(Image(filename=g.render(filename='img/automaton')))


if __name__ == "__main__":
    pass
