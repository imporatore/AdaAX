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
* forbid the transition from going to the accepting state too early
  * forbid merging into the accepting state
  * add threshold for merging F
* sampling


### Problems
* Should the start & accepting states attend in the merging procedure?