import argparse
import os
import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

torch.manual_seed(1)

from src.NRE.models import NRE
from src.NRE.utils import load_pretrained_embedding, load_data, batch_loader

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return Variable(tensor)


def prepare_vocab(sequence, w2i = {}):

    for w in sequence:
        if w not in w2i:
            w2i[w] = len(w2i)

    return w2i


def train(args):

    # import ipdb;ipdb.set_trace()
    trainX, trainY = load_data(os.path.join(args.data_dir, "train.txt"))
    devX, devY = load_data(os.path.join(args.data_dir, "dev.txt"))

    w2i = {"_UNK_": 0, "_PAD_": 1}
    word2idx = prepare_vocab([w for i, j in trainX+devX for w in i+j], w2i = w2i)

    t2i = {"NONE": 0}
    tag2idx = prepare_vocab([l for l in trainY+devY], w2i = t2i)
    num_classes = len(tag2idx)

    n_batches = len(trainX) // args.batch_size
    dev_n_batches = len(devX) // args.batch_size

    model = NRE(args.embed_dim,
                args.hidden_dim,
                vocab_size=len(word2idx),
                tagset_size=len(tag2idx))
    if args.cuda:
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in tqdm.trange(300, ncols=100, desc="Epoch"):
        epoch_loss = 0
        correct = 0
        train_batches = batch_loader(inputs=(trainX, word2idx),
                                     targets=(trainY, tag2idx),
                                     batch_size=args.batch_size,
                                     cuda=args.cuda)
        dev_batches = batch_loader(inputs=(devX, word2idx),
                                   targets=(devY, tag2idx),
                                   batch_size=args.batch_size,
                                   cuda=args.cuda)
        # Train
        model.train()
        for ps, p_lens, cs, c_lens, ys in tqdm.tqdm(train_batches,
                                                    total=n_batches,
                                                    ncols=100,
                                                    desc="Training"):

            optimizer.zero_grad()
            predictions = model(ps, p_lens, cs, c_lens)
            loss = loss_function(predictions, ys)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]
            correct += torch.sum(torch.eq(predictions.data.max(dim=1)[1],
                                          ys.data))
        # TODO: F-1
        acc = correct / len(trainX)

        # Dev
        # Compute F-1.
        TP = [0 for _ in range(num_classes)]
        FP = [0 for _ in range(num_classes)]
        FN = [0 for _ in range(num_classes)]

        model.eval()
        correct = 0
        eval_loss = 0
        for ps, p_lens, cs, c_lens, ys in tqdm.tqdm(dev_batches,
                                                    total=dev_n_batches,
                                                    ncols=100,
                                                    desc="Evaluating"):
            predictions = model(ps, p_lens, cs, c_lens)
            loss = loss_function(predictions, ys)
            eval_loss += loss.data[0]
            preds = predictions.data.max(dim=1)[1]
            correct += torch.sum(torch.eq(preds, ys.data))

            for pred, y in zip(preds, ys.data):
                if pred == y:
                    TP[y] += 1
                else:
                    FP[pred] += 1
                    FN[y] += 1

        P = [ float(tp)/(tp+fp) if tp+fp > 0 else 0 for tp,fp in zip(TP, FP)]
        R = [ float(tp)/(tp+fn) if tp+fn > 0 else 0 for tp,fn in zip(TP, FN)]
        F1 = [ 2*p*r/(p+r) if p+r > 0 else 0 for p,r in zip(P, R)]
        macro_F1 = np.mean(F1[1:])
        print("F1  :",F1)
        print("TP  :",TP)
        print("FP  :",FP)
        print("FN  :", FN)
        print("P  :", P)
        print("R  :", R)
        print("Macro-Average F1: {}".format(macro_F1))
        eval_acc = correct / len(devX)
        tqdm.tqdm.write(("Epoch {:2d}\ttr_loss/acc: ({:.3f} | {:.2f})"
                         "\tdv_loss/acc: ({:.3f} | {:.2f})").format(
            epoch+1, epoch_loss, acc, eval_loss, eval_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Relation classifier")
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-file", type=str)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    train(args)
