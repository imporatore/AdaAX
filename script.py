from config import DFA_DIR, IMAGE_DIR, POS_THRESHOLD, SAMPLE_THRESHOLD, TAU, DELTA
from Helpers import ConfigDict
from AdaAX import main

DEFAULT_CONFIG = {"dfa_dir": DFA_DIR,
                  "image_dir": IMAGE_DIR,
                  "add_single_sample": False,
                  "plot": True}

synthetic1_config = {"fname": "synthetic_data_1",
                     "pos_threshold": .95,
                     "sample_threshold": 5,
                     "neighbour": 1.,
                     "fidelity_loss": 0.,
                     "absorb": True,
                     "class_balanced": False,
                     "merge_start": True,
                     "merge_accept": True,
                     "search": 'first'}

synthetic2_config = {"fname": "synthetic_data_2",
                     "pos_threshold": .95,
                     "sample_threshold": 5,
                     "neighbour": 1.5,
                     "fidelity_loss": 0.,
                     "absorb": True,
                     "class_balanced": False,
                     "merge_start": True,
                     "merge_accept": True,
                     "search": 'first'}

tomita1_config = {"fname": "tomita_data_1",
                  "pos_threshold": .95,
                  "sample_threshold": 2,
                  "neighbour": 0.5,
                  "fidelity_loss": 0.,
                  "absorb": False,
                  "class_balanced": False,
                  "merge_start": True,
                  "merge_accept": True,
                  "search": 'first'}

tomita2_config = {}

yelp_config = {"fname": "yelp_review_balanced",
               "pos_threshold": POS_THRESHOLD,
               "sample_threshold": SAMPLE_THRESHOLD,
               "neighbour": TAU,
               "fidelity_loss": DELTA,
               "absorb": False,
               "class_balanced": False,
               "merge_start": True,
               "merge_accept": True,
               "search": 'best'}


def run(name, type, config=None):
    cfg = ConfigDict(DEFAULT_CONFIG)

    if name == 'synthetic1':
        cfg.update(synthetic1_config)
    elif name == 'synthetic2':
        cfg.update(synthetic2_config)
    elif name == 'tomita1':
        cfg.update(tomita1_config)
    elif name == 'tomita2':
        cfg.update(tomita2_config)
    elif name == 'yelp':
        cfg.update(yelp_config)
    else:
        raise ValueError('Task %s not found.' % name)

    if type == 'rnn':
        cfg.update({"model": 'rnn'})
    elif type == 'lstm':
        cfg.update({"model": 'lstm'})
    elif type == 'gru':
        cfg.update({"model": 'gru'})
    else:
        raise ValueError('Model type %s not found.' % type)

    if config:
        cfg.update(config)

    main(cfg)


if __name__ == "__main__":
    # run('synthetic1', 'rnn')  # todo: MINOR MISTAKE (seems RNN mistake)
    # run('synthetic1', 'lstm')
    # run('synthetic1', 'gru')
    # run('synthetic2', 'rnn')  # todo: MINOR MISTAKE
    # run('synthetic2', 'lstm')
    # run('synthetic2', 'gru')
    # run('tomita1', 'rnn')
    # run('tomita1', 'lstm', {"neighbour": 5.})
    run('tomita1', 'gru', {"neighbour": 5.})
    # run('tomita2', 'rnn')
    # run('tomita2', 'lstm')
    # run('tomita2', 'gru')
    # run('yelp', 'rnn')
    # run('yelp', 'lstm')
    # run('yelp', 'gru')
    pass
