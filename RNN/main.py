import argparse
import os

import torch
import torch.nn as nn
import pandas as pd

from tqdm import tqdm, trange

from config import RANDOM_STATE, RNN_MODEL_DIR, RNN_RESULT_DIR
from data.utils import save2csv, save2npy
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
        train_data = tqdm(train_data)
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
        print("training accuracy: {}/{} ".format(correct, total), correct / total)
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
        train_data = tqdm(train_data)
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
        print("\ntraining accuracy: {}/{} ".format(correct, total), correct / total)

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
    print("validation accuracy: {}/{} ".format(correct, total), correct / total)
    return correct / total


def predict(model, dataloader):
    with torch.no_grad():
        model.eval()
        prediction = []
        for i, (seq, _) in enumerate(tqdm(dataloader)):

            # convert to cuda
            seq = seq.to(device)

            output, hidden = model(seq)
            output = torch.sigmoid(output).round().cpu().numpy()
            prediction.extend(output.tolist())

    return prediction


def predict_and_save(save_path, fname, model, dataloader):
    with torch.no_grad():
        model.eval()
        prediction = []
        for i, (seq, label) in enumerate(tqdm(dataloader)):

            # convert to cuda
            seq = seq.to(device)

            output, hidden = model(seq)
            seq = seq.cpu().numpy()
            hidden = hidden.cpu().numpy()
            try:
                label = label.cpu().numpy().view(-1, 1)
            except ValueError:
                label = label.cpu().numpy()
            output = torch.sigmoid(output).round().cpu().numpy()
            prediction.extend(output.tolist())

            # Package data
            data = {
                'input': seq,
                'hidden': hidden,
                'labels': label,
                'predictions': output
            }

            save2npy(save_path, data, f"{fname}_data_{i}.npy")

    return prediction


def main(config):
    # create model directory
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    # load data
    train_data, valid_data, test_data, vocab = get_loader(
        fname=config.fname,
        batch_size=config.batch_size
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

    print("\nmodel loaded...")

    if valid_data:
        train_and_validate(model=model,
                           model_path=os.path.join(config.model_dir, "{}_{}.pkl".format(config.fname, config.model)),
                           train_data=train_data,
                           valid_data=valid_data,
                           learning_rate=config.learning_rate,
                           total_epoch=config.total_epoch)
    else:
        train(model=model,
              model_path=os.path.join(config.model_dir, "{}_{}.pkl".format(config.fname, config.model)),
              train_data=train_data,
              learning_rate=config.learning_rate,
              total_epoch=config.total_epoch)

    # model.load_state_dict(torch.load(os.path.join(config.model_dir, "{}_{}.pkl".format(config.fname, config.model))))

    train_output = predict_and_save(save_path=config.result_dir,
                                    fname="{}_{}_train".format(config.fname, config.model),
                                    model=model,
                                    dataloader=train_data)
    if valid_data:
        valid_output = predict_and_save(save_path=config.result_dir,
                                        fname="{}_{}_valid".format(config.fname, config.model),
                                        model=model,
                                        dataloader=valid_data)
    test_output = predict_and_save(save_path=config.result_dir,
                                   fname="{}_{}_test".format(config.fname, config.model),
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
    save2csv(config.result_dir, sub_df, "{}_{}_test_predict.csv".format(config.fname, config.model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # setup parameters
    parser.add_argument("--fname", type=str, default="yelp_review_balanced")
    parser.add_argument("--model", type=str, default="rnn", choices=["rnn", "lstm", "gru", "glove-lstm"])
    parser.add_argument("--model_dir", type=str, default=RNN_MODEL_DIR)
    parser.add_argument("--result_dir", type=str, default=RNN_RESULT_DIR)
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
