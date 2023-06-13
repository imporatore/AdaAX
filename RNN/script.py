from config import RNN_MODEL_DIR, RNN_RESULT_DIR, START_SYMBOL
from utils import ConfigDict
from main import main


DEFAULT_CONFIG = {"model_dir": RNN_MODEL_DIR,
                  "result_dir": RNN_RESULT_DIR,
                  "dropout_rate": 0.5,
                  "batch_size": 100,
                  # "start_symbol": START_SYMBOL,
                  "load_vocab": False,
                  "save_vocab": True,
                  "load_loader": False,
                  "save_loader": True,
                  "load_model": False,
                  "need_train": True}

synthetic1_config = {"fname": "synthetic_data_1",
                     "hidden_size": 16,
                     "embedding_size": 16,
                     "total_epoch": 5,  # 10 for RNN
                     "learning_rate": 0.01}

synthetic2_config = {"fname": "synthetic_data_2",
                     "hidden_size": 16,
                     "embedding_size": 16,
                     "total_epoch": 10,  # 5
                     "learning_rate": 0.01}  # 0.005

tomita1_config = {"fname": "tomita_data_1",
                  "hidden_size": 32,
                  "embedding_size": 32,
                  "total_epoch": 5,  # 10 for RNN
                  "learning_rate": 0.005}

tomita2_config = {}

yelp_config = {"fname": "yelp_review_balanced",
               "hidden_size": 256,
               "embedding_size": 300,
               "total_epoch": 30,
               "learning_rate": 0.001}


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
    # run('synthetic1', 'rnn', {"load_vocab": True, "load_loader": True, "total_epoch": 10})
    # run('synthetic1', 'lstm', {"load_vocab": True, "load_loader": True})
    # # run('synthetic1', 'gru', {"load_vocab": True, "load_loader": True})
    # run('synthetic2', 'rnn', {"load_vocab": True, "load_loader": True})
    # run('synthetic2', 'lstm', {"load_vocab": True, "load_loader": True, "total_epoch": 10, "learning_rate": 0.005})
    # run('synthetic2', 'gru', {"load_vocab": True, "load_loader": True, "total_epoch": 10, "learning_rate": 0.005})
    run('tomita1', 'rnn', {"load_vocab": True, "load_loader": True, "total_epoch": 10})
    run('tomita1', 'lstm', {"load_vocab": True, "load_loader": True})
    run('tomita1', 'gru', {"load_vocab": True, "load_loader": True})
    run('yelp', 'rnn', {"load_vocab": True, "load_loader": True})
    run('yelp', 'lstm', {"load_vocab": True, "load_loader": True})
    run('yelp', 'gru', {"load_vocab": True, "load_loader": True})
    # run('tomita1', 'gru', {"load_vocab": True, "load_loader": True})
    # run('yelp', 'lstm', {"load_vocab": True, "load_loader": True})
    # run('yelp', 'gru', {"load_vocab": True, "load_loader": True})
