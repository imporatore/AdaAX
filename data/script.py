import random

from config import DATA_DIR
from data.Grammars import *
from data.utils import gen_dataset, save2pickle, save2csv

random.seed(13579)


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

    # yelp dataset from kaggle
    df = pd.read_json(os.path.join(DATA_DIR, "yelp_academic_dataset_review.json"), lines=True)
    df = df[['stars', 'text']]  # keep only the 'stars' and 'text' columns

    # convert into binary label, 4-5 stars for 1 and 1-3 stars for 0
    df['label'] = df['stars'].apply(lambda rating: 1 if rating > 3 else 0)
    df = df[['text', 'label']]  # Keep only the 'text' and 'sentiment' columns

    save2csv(DATA_DIR, df, 'yelp_review')

