from collections import Counter

import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence


# todo: use pretrained word embeddings?
def preprocess(data, target, start_symbol=None, tokenizer=None):
    counter = Counter()
    if isinstance(data, list):  # synthetic data or tomita data
        if start_symbol:
            data = [start_symbol + dat for dat in data]

        def tokenizer(expr):
            return list(expr)

    else:  # real-word dataset in csv form
        tokenizer = get_tokenizer(tokenizer, language='en')

    for dat in data:
        counter.update(tokenizer(dat))

    vocab = Vocab(counter, min_freq=1)  # Create vocab

    def data_process(raw_data_iter):
        data = [torch.tensor([vocab[token] for token in tokenizer(item[1])], dtype=torch.long) for item in
                raw_data_iter]
        data = pad_sequence(data)
        target = torch.tensor([item[0] - 1 for item in raw_data_iter], dtype=torch.long)
        return data, target

    # Create DataLoader
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=False, collate_fn=data_process)
