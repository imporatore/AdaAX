* Raw Data
  - train.csv, test.csv
    
    **Yelp polarity review** dataset downloaded from [yelp-polarity-review-dataset](https://metatext.io/datasets/yelp-polarity-reviews).
    
  - EPM (todo)
  
    **Educational Process Mining** dataset downloaded from [UCI](https://archive.ics.uci.edu/ml/datasets/Educational+Process+Mining+%28EPM%29%3A+A+Learning+Analytics+Data+Set#).
  
* Data
  
  - synthetic data
  
    Binary vector with elements randomly drawn from $\{0, 1\}$, $20000$ samples with fixed length $15$.
  
    - synthetic_data_1.pickle
  
    - synthetic_data_2.pickle
  
  - Tomita grammar
  
    Class-balanced samples with variable lengths $[0, 1, 2, \dots,15, 20, 25, 30]$ were generated. For each length, 5000 samples were searched.
  
    - tomita_data_1.pickle
  
      Tomita 4: contains consecutive subsequence “000”
  
    - tomita_data_2.pickle
  
      Tomita 7: regular expression *“(0) + (1) + (0) + (1)”*
  
  - yelp_review_balanced.csv
  
    Converted into binary labels, 4-5 stars for positive (1) and 1-3 stars for negative (0). 20000 class balanced samples.
  
* .py
  - script.py
  
    run *script.py* to create processed data
  
  - Grammars.py
  
    - Tomita grammars
    - Ground truth for synthetic data:
      * contains consecutive subsequence *“11111”*
      * regular expression *“(1)+0(1)+01”*
  
  - utils.py
