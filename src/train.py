import argparse
import os
import sys
import tqdm
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(1)

from src.NRE.models import NRE
from src.NRE.utils import (load_pretrained_embedding,
                           load_data,
                           batch_loader,
                           save_checkpoint)


def prepare_vocab(sequence, w2i={}):
    """Calculate dictionary."""
    for w in sequence:
        if w not in w2i:
            w2i[w] = len(w2i)

    return w2i


def train(args):

    if os.path.exists(args.save_dir):
        print("Make a new save directory.")
        sys.exit(0)
    else:
        os.mkdir(args.save_dir)

    trainX, trainY = load_data(os.path.join(args.data_dir, "train.txt"))
    devX, devY = load_data(os.path.join(args.data_dir, "dev.txt"))

    print("Tr: {}, Dv: {}".format(len(trainX), len(devX)))

    w2i = {"_UNK_": 0, "_PAD_": 1}
    word2idx = prepare_vocab([w for i, j in trainX+devX for w in i+j], w2i=w2i)

    t2i = {"NONE": 0}
    tag2idx = prepare_vocab([l for l in trainY+devY], w2i=t2i)
    num_classes = len(tag2idx)

    if args.embedding_file:
        W, embed_dim = load_pretrained_embedding(w2i, args.embedding_file)
        assert embed_dim == args.embed_dim
    else:
        W = None

    model = NRE(args.embed_dim,
                args.hidden_dim,
                vocab_size=len(word2idx),
                tagset_size=len(tag2idx),
                pretrained_emb=W,
                freeze_emb=args.freeze_emb)
    if args.cuda:
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr)

    n_batches = len(trainX) // args.batch_size
    dev_n_batches = len(devX) // args.batch_size

    with open(os.path.join(args.save_dir, "train.log"), "a") as f:
        print("\t\t".join(["Epoch", "Tr loss", "Dv loss", "Macro-F1"]),
              file=f)

    macro_F1_best = 0    # For tracking best macro F1
    F1_best = None       # For per-class F1
    tolerance_count = 0  # For tracking the number of waits

    for epoch in tqdm.trange(args.num_epochs, ncols=100, desc="Epoch"):
        epoch_loss = 0
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

        # Dev
        # Compute F-1.
        TP = [0 for _ in range(num_classes)]
        FP = [0 for _ in range(num_classes)]
        FN = [0 for _ in range(num_classes)]

        model.eval()
        eval_loss = 0
        for ps, p_lens, cs, c_lens, ys in tqdm.tqdm(dev_batches,
                                                    total=dev_n_batches,
                                                    ncols=100,
                                                    desc="Evaluating"):
            predictions = model(ps, p_lens, cs, c_lens)
            loss = loss_function(predictions, ys)
            eval_loss += loss.data[0]
            preds = predictions.data.max(dim=1)[1]

            for pred, y in zip(preds, ys.data):
                if pred == y:
                    TP[y] += 1
                else:
                    FP[pred] += 1
                    FN[y] += 1

        P = [float(tp)/(tp+fp) if tp+fp > 0 else 0 for tp, fp in zip(TP, FP)]
        R = [float(tp)/(tp+fn) if tp+fn > 0 else 0 for tp, fn in zip(TP, FN)]
        F1 = [2*p*r/(p+r) if p+r > 0 else 0 for p, r in zip(P, R)]
        macro_F1 = np.mean(F1[1:])

        tqdm.tqdm.write(("Epoch {:2d}\ttr_loss: {:.3f}"
                         "\tdv_loss / F1: ({:.3f} | {:.3f})").format(
                             epoch+1, epoch_loss, eval_loss, macro_F1))

        with open(os.path.join(args.save_dir, "train.log"), "a") as f:
            print("{:2d}\t\t{:3.2f}\t\t{:.2f}\t\t{:.3f}".format(
                epoch+1, epoch_loss, eval_loss, macro_F1),
                  file=f)

        if macro_F1 > macro_F1_best:
            save_checkpoint(model.state_dict(),
                            False,
                            os.path.join(args.save_dir, "model_best.pt"))
            macro_F1_best = macro_F1
            F1_best = F1
            tolerance_count = 0

        elif tolerance_count > args.tolerate:
            break

        else:
            tolerance_count += 1

    print("Best Macro F1: {}".format(macro_F1_best))
    tags = sorted(tag2idx.items(), key=lambda x: x[1])
    print(tabulate([[t[0], f] for t, f in zip(tags, F1_best)],
                   headers=("Tag", "F1")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Relation classifier")
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--embed-dim", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--embedding-file", type=str,
                        help="Path to pretrained embeddings.")
    parser.add_argument("--freeze-emb", action="store_true", default=False)
    parser.add_argument("--tolerate", type=int, default=5,
                        help="# of epochs to wait when the metric gets worse.")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--lr", type=int, default=0.1)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save the experiment.")
    args = parser.parse_args()

    train(args)
