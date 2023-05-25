import argparse
import os

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from tqdm import tqdm, trange

from config import RANDOM_STATE, RNN_MODEL_DIR, RNN_RESULT_DIR, START_SYMBOL
from data.utils import save2csv, save2npy, save2pickle
from RNN.data_loader import get_loader
from RNN.model import VanillaRNN, VanillaLSTMModel, VanillaGRUModel, GloveModel

# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(RANDOM_STATE)


def train_and_validate(model, model_path, train_data, valid_data, learning_rate, total_epoch):

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    best_acc, hits, train_loss = 0, 0, 0.0

    for epoch in trange(total_epoch):
        correct, total = 0, 0
        # train_data = tqdm(train_data)
        model.train()

        if hits == 3:
            print("switching to ASGD")
            optimizer = torch.optim.ASGD(model.parameters(), lr=learning_rate)

        for i, (seq, label) in enumerate(train_data):

            seq = seq.to(device)
            label = label.to(device).view(-1, 1).float()

            optimizer.zero_grad()
            output, _ = model(seq)
            loss = criterion(output, label)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

            output = torch.sigmoid(output).round()
            correct += (output == label).sum().item()
            total += output.size(0)

        # print epoch accuracy
        print("\nEpoch {} training accuracy: {}/{} ".format(epoch + 1, correct, total), correct / total)
        valid_acc = validate(model, valid_data)

        # keep best acc model
        if valid_acc > best_acc:
            torch.save(model.state_dict(), model_path)
            best_acc, hits = valid_acc, 0
        else:
            hits += 1


def train(model, model_path, train_data, learning_rate, total_epoch):

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in trange(total_epoch):
        correct, total = 0, 0
        # train_data = tqdm(train_data)
        for i, (seq, label) in enumerate(train_data):

            seq = seq.to(device)
            label = label.to(device).view(-1, 1).float()

            optimizer.zero_grad()
            output, _ = model(seq)
            loss = criterion(output, label)

            loss.backward()
            optimizer.step()

            output = torch.sigmoid(output).round()
            correct += (output == label).sum().item()
            total += output.size(0)

        # print epoch accuracy
        print("\nEpoch {} training accuracy: {}/{} ".format(epoch + 1, correct, total), correct / total)

    torch.save(model.state_dict(), model_path)


def validate(model, valid_data):
    # turn on testing behavior for dropout, batch-norm
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        for i, (seq, label) in enumerate(valid_data):

            # convert to cuda
            seq = seq.to(device)
            label = label.to(device).view(-1, 1)

            # Forward
            output, _ = model(seq)
            output = torch.sigmoid(output).round()

            correct += (output == label).sum().item()
            total += output.size(0)
    print("Validation accuracy: {}/{} ".format(correct, total), correct / total)
    return correct / total


def predict(model, dataloader):
    with torch.no_grad():
        model.eval()
        prediction = []
        for i, (seq, _) in enumerate(tqdm(dataloader)):

            # convert to cuda
            seq = seq.to(device)

            output, _ = model(seq)
            output = torch.sigmoid(output).round().cpu().numpy()
            prediction.extend(output.tolist())

    return prediction


def predict_and_save(save_path, fname, model, dataloader):
    with torch.no_grad():
        model.eval()
        prediction, res = [], {}
        for i, (seq, label) in enumerate(tqdm(dataloader)):

            # convert to cuda
            seq = seq.to(device)

            output, hidden = model(seq)
            seq = seq.cpu().numpy()
            hidden = hidden.cpu().numpy()
            label = label.cpu().numpy()
            output = torch.sigmoid(output).cpu().numpy().squeeze()
            prediction.extend(output.round().tolist())

            # Package data
            res['input'] = np.concatenate((res.get('input', np.ndarray(shape=(0, *seq.shape[1:]))), seq), axis=0)
            res['hidden'] = np.concatenate((res.get('hidden', np.ndarray(shape=(0, *hidden.shape[1:]))), hidden), axis=0)
            res['output'] = np.concatenate((res.get('output', np.ndarray(shape=(0, *output.shape[1:]))), output), axis=0)
            res['labels'] = np.concatenate((res.get('labels', np.ndarray(shape=(0, *label.shape[1:]))), label), axis=0)

    save2npy(save_path, res, f"{fname}_data")

    return prediction


def main(config):
    # create model directory
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    model_dir = os.path.join(config.model_dir, "{}_{}.pkl".format(config.fname, config.model))
    result_dir = os.path.join(config.result_dir, config.fname, config.model)

    # load data
    train_data, valid_data, test_data, vocab = get_loader(
        fname=config.fname,
        batch_size=config.batch_size,
        start_symbol=config.start_symbol,
        load_vocab=config.load_vocab,
        save_vocab=config.load_vocab,
        load_loader=config.load_loader,
        save_loader=config.save_loader
    )

    # model
    if config.model == "rnn":
        model = VanillaRNN(config.embedding_size,
                           config.hidden_size,
                           config.dropout_rate,
                           vocab).to(device)

    elif config.model == "lstm":
        model = VanillaLSTMModel(config.embedding_size,
                                 config.hidden_size,
                                 config.dropout_rate,
                                 vocab).to(device)

    elif config.model == "gru":
        model = VanillaGRUModel(config.embedding_size,
                                config.hidden_size,
                                config.dropout_rate,
                                vocab).to(device)

    elif config.model == 'glove-lstm':
        glove = vocab.get_embedding('glove', config.embedding_size)
        model = GloveModel(config.embedding_size,
                           config.hidden_size,
                           config.dropout_rate,
                           glove).to(device)

    print("\n{} model for {} loaded...".format(config.model, config.fname))

    if config.load_model:
        model.load_state_dict(torch.load(model_dir))

    if config.need_train:
        if valid_data:
            train_and_validate(model=model,
                               model_path=model_dir,
                               train_data=train_data,
                               valid_data=valid_data,
                               learning_rate=config.learning_rate,
                               total_epoch=config.total_epoch)
        else:
            train(model=model,
                  model_path=model_dir,
                  train_data=train_data,
                  learning_rate=config.learning_rate,
                  total_epoch=config.total_epoch)

    train_output = predict_and_save(save_path=result_dir,
                                    fname="train",
                                    model=model,
                                    dataloader=train_data)
    if valid_data:
        valid_output = predict_and_save(save_path=result_dir,
                                        fname="valid",
                                        model=model,
                                        dataloader=valid_data)
    test_output = predict_and_save(save_path=result_dir,
                                   fname="test",
                                   model=model,
                                   dataloader=test_data)
    sub_df = pd.DataFrame()
    try:
        sub_df['text'] = test_data.dataset.df['text']
        sub_df['label'] = test_data.dataset.df['label']
    except AttributeError:
        sub_df['expr'] = [dat[0] for dat in test_data.dataset.data]
        sub_df['label'] = [dat[1] for dat in test_data.dataset.data]
    sub_df["rnn_predict"] = test_output
    save2csv(result_dir, sub_df, "test_predict")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup parameters
    parser.add_argument("--fname", type=str, default="yelp_review_balanced")
    parser.add_argument("--model", type=str, default="rnn", choices=["rnn", "lstm", "gru", "glove-lstm"])
    parser.add_argument("--model_dir", type=str, default=RNN_MODEL_DIR)
    parser.add_argument("--result_dir", type=str, default=RNN_RESULT_DIR)
    parser.add_argument("--start_symbol", type=str, default=START_SYMBOL)
    parser.add_argument("--load_vocab", type=bool, default=True)
    parser.add_argument("--save_vocab", type=bool, default=True)
    parser.add_argument("--load_loader", type=bool, default=True)
    parser.add_argument("--save_loader", type=bool, default=True)
    # parser.add_argument("--load_data", type=bool, default=True)
    parser.add_argument("--load_model", type=bool, default=False)
    parser.add_argument("--need_train", type=bool, default=True)
    # parser.add_argument("--vocab_path", type=str, default='vocab.pkl')

    # model parameters
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--embedding_size", type=int, default=100)
    parser.add_argument("--dropout_rate", type=float, default=0.2)

    # training parameters
    parser.add_argument("--total_epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    # parser.add_argument("--logging_rate", type=int, default=100)
    args = parser.parse_args()

    print(args)
    print('device: ', device)

    main(args)
