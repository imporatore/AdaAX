**Two types of implementation**
* Easy mode:
    * use index and level to find corresponding hidden states and symbols
    * use 
* Complex mode:
  * use data structure

# Is pattern extraction & build DFA split?

core set/pure set/focal set
pure
prefix unique
suffix same

### Examine
* is transition determined?
* is the extracted pattern unique?


### START
* use start symbol
* represent if as empty list

### Twisting
* forbid the transition from going to the accept state too early
  * forbid merging into the accept state
  * add threshold for merging F
* sampling (data flow)


### Problems
* Should the start & accept states attend in the merging procedure?
* the order of patterns added to the DFA affects?

### Alphabet
* one-hot
* word embedding?

### todo
* pre-clustering method

### more
RNN hidden states help with clustering and merging. Helps to filter out trivial pattern and merge similar states.
The pre-pruned DFA is based on patterns(symbols). Quite different from pre-clustering techniques when we use only the RNN info.
* add validation set to determine merging and threshold for pruning

### Question
* hidden states are backtracked only for the positive samples, what for the states which leads to negative samples?
  * if attended in the clustering period, are they remained as a cluster? the transition to them?
* How to deal with missing transitions? Do they exist?


* cashed_property
* * bugs in start symbol