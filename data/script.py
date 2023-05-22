import random

from config import DATA_DIR, RANDOM_STATE
from data.Grammars import *
from data.utils import gen_dataset, save2pickle, save2csv

random.seed(RANDOM_STATE)


def gen_synthetic_dataset(path=DATA_DIR, ftype='pickle'):
    synthetic_data_1 = gen_dataset("01", rule1, 15, 20000, class_balance=False)
    synthetic_data_2 = gen_dataset("01", rule2, 15, 20000, class_balance=False)
    if ftype == 'pickle':
        save2pickle(path, synthetic_data_1, "synthetic_data_1")
        save2pickle(path, synthetic_data_2, "synthetic_data_2")
        return
    elif ftype in ['pd', 'pandas', 'csv']:
        save2csv(path, synthetic_data_1, "synthetic_data_1")
        save2csv(path, synthetic_data_2, "synthetic_data_2")
        return
    raise ValueError("Argument ftype should be either 'pickle' or 'pandas'.")


def gen_tomita_dataset(path=DATA_DIR, ftype='pickle'):
    tomita_data_1 = gen_dataset("01", tomita4, list(range(15)) + [15, 20, 25, 30], 5000)
    tomita_data_2 = gen_dataset("01", tomita7, list(range(15)) + [15, 20, 25, 30], 5000)
    if ftype == 'pickle':
        save2pickle(path, tomita_data_1, "tomita_data_1")
        save2pickle(path, tomita_data_2, "tomita_data_2")
        return
    elif ftype in ['pd', 'pandas', 'csv']:
        save2csv(path, tomita_data_1, "tomita_data_1")
        save2csv(path, tomita_data_2, "tomita_data_2")
        return
    raise ValueError("Argument ftype should be either 'pickle' or 'pandas'.")


if __name__ == "__main__":
    import os
    import pandas as pd

    gen_synthetic_dataset()
    gen_tomita_dataset()

    # yelp review polarity dataset
    yelp_train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), header=None)
    yelp_test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"), header=None)

    yelp_train_df.columns, yelp_test_df.columns = ['stars', 'expr'], ['stars', 'expr']  # name columns
    yelp_review_df = pd.concat([yelp_train_df, yelp_test_df])  # merge two dataset

    # convert into binary label, 4-5 stars for 1 and 1-3 stars for 0
    yelp_review_df['label'] = yelp_review_df['stars'].apply(lambda rating: 1 if rating > 3 else 0)

    yelp_review_df['len'] = yelp_review_df['expr'].apply(lambda string: len(' '.split(string)))
    # # select only reviews that are less than 25 words and the 'expr' and 'label' columns
    yelp_review_df = yelp_review_df.loc[yelp_review_df['len'] < 25, ['expr', 'label']]

    # generate 20000 class-balanced sample
    yelp_pos_review = yelp_review_df.loc[yelp_review_df['label'] == 1].reset_index(drop=True)
    yelp_neg_review = yelp_review_df.loc[yelp_review_df['label'] == 0].reset_index(drop=True)
    yelp_pos_review = yelp_pos_review.sample(10000, random_state=RANDOM_STATE)
    yelp_neg_review = yelp_neg_review.sample(10000, random_state=RANDOM_STATE)

    yelp_review_balanced = pd.concat([yelp_pos_review, yelp_neg_review])
    yelp_review_balanced = yelp_review_balanced.sample(20000)

    save2csv(DATA_DIR, yelp_review_balanced, 'yelp_review_balanced')
