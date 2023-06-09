from config import DFA_DIR, IMAGE_DIR, K, THETA, TAU, DELTA
from utils import ConfigDict
from AdaAX import main


DEFAULT_CONFIG = {"dfa_dir": DFA_DIR,
                  "image_dir": IMAGE_DIR,
                  # "start_symbol": START_SYMBOL,
                  "plot": True}

synthetic1_config = {"fname": "synthetic_data_1",
                     "clusters": K,
                     "pruning": THETA,
                     "merge_start": True,
                     "merge_accept": True,
                     "neighbour": TAU,
                     "fidelity_loss": DELTA}

synthetic2_config = {"fname": "synthetic_data_2",
                     "clusters": K,
                     "pruning": THETA,
                     "merge_start": True,
                     "merge_accept": True,
                     "neighbour": TAU,
                     "fidelity_loss": DELTA}

tomita1_config = {"fname": "tomita_data_1",
                  "clusters": K,
                  "pruning": THETA,
                  "merge_start": True,
                  "merge_accept": True,
                  "neighbour": TAU,
                  "fidelity_loss": DELTA}

tomita2_config = {"fname": "tomita_data_2",
                  "clusters": K,
                  "pruning": THETA,
                  "merge_start": True,
                  "merge_accept": True,
                  "neighbour": TAU,
                  "fidelity_loss": DELTA}

yelp_config = {"fname": "yelp_review_balanced",
               "clusters": K,
               "pruning": THETA,
               "merge_start": True,
               "merge_accept": True,
               "neighbour": TAU,
               "fidelity_loss": DELTA}


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
    # names = ['synthetic1', 'synthetic2', 'tomita1', 'tomita2', 'yelp']
    # models = ['rnn', 'lstm', 'gru']
    # for nam in names:
    #     for mod in models:
    #         run(nam, mod)
    run('synthetic1', 'gru', {"load_vocab": True, "load_loader": True})
    run('tomita1', 'gru', {"load_vocab": True, "load_loader": True})
    run('yelp', 'lstm', {"load_vocab": True, "load_loader": True})
    run('yelp', 'gru', {"load_vocab": True, "load_loader": True})
