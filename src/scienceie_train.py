import sys
from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
color_scheme='Linux', call_pdb=1)


import argparse
import os
import sys
import tqdm
import numpy as np
import pickle
from collections import defaultdict
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

np.random.seed(1)
torch.manual_seed(1)

from src.CNN.models import CNNclassifier
from src.CNN.models import LSTMRelationClassifier
from src.NRE.utils import load_pretrained_embedding, save_checkpoint


def pad(features, max_seq_len, single_ex=False):
    # Padding idx = 1
    if single_ex:
        return [f + [1]*(max_seq_len - len(f)) for f in features]

    return [[f + [1]*(max_seq_len - len(f)) for f in r]
            for r in features]

def denum(dicts, r):
    r = r.cpu().tolist()
    denumed = []
    anno = ["word", "relpos", "relpos", "ner", "pos"]
    for i, feat in enumerate(r):
        denumed.append([dicts[anno[i]][1][j] for j in feat if j != 1])
    return denumed


def upsample(data, upsample_ratio):
    pos = [i for i in data if i[1] != 0]
    neg = [i for i in data if i[1] == 0]
    upsamples = len(neg) * upsample_ratio - len(pos)
    duplicate_idx = np.random.choice(range(len(pos)), int(upsamples))
    return data+[pos[int(i)] for i in duplicate_idx]


def load_data(path):
    # Load data
    with open(os.path.join(path, "data.pkl"), "rb") as f:
        data = pickle.load(f)

    # Load dict
    with open(os.path.join(path, "dict.pkl"), "rb") as f:
        dicts = pickle.load(f)

    return data, dicts


def aggregate_stats(labvocab, stats):
    agg = defaultdict(int)
    for rel, val in zip(labvocab, stats):
        if rel.startswith("r_"):
            agg[rel[2:]] += val
        else:
            agg[rel] += val
    # Force None to be at the head
    return ([agg["None"]] + [v for r, v in agg.items() if r != "None"],
            ["None"] + [k for k in agg.keys() if k != "None"])


def batch_loader(data, dist1id, ent0id, batch_size=1, cuda=True):
    np.random.shuffle(data)

    n_batch_size = len(data) // batch_size

    for i in range(n_batch_size):
        rs, targets = zip(*data[batch_size*i:batch_size*(i+1)])

        words = [r[0] for r in rs]
        word_lengths = [len(w) for w in words]

        max_seq_len = max(word_lengths)
        rs = pad(rs, max_seq_len)

        locs1 = [r[1] for r in rs]
        locs2 = [r[2] for r in rs]
        p_end =  [loc.index(dist1id)
                  if dist1id in loc
                  else len(loc) - loc[::-1].index(ent0id) for loc in locs1]
        c_start = [loc.index(ent0id) for loc in locs2]

        ps = [r[0][:end] for r, end in zip(rs, p_end)]
        cs = [r[0][start:] for r, start in zip(rs, c_start)]

        ps_lengths = [len(p) for p  in ps]
        cs_lengths = [len(c) for c in cs]

        max_p_len = max(ps_lengths)
        max_c_len = max(cs_lengths)

        ps = pad(ps, max_p_len, single_ex=True)
        cs = pad(cs, max_c_len, single_ex=True)

        ps = torch.LongTensor(ps)
        cs = torch.LongTensor(cs)
        ps_lengths = torch.LongTensor(ps_lengths)
        cs_lengths = torch.LongTensor(cs_lengths)
        word_lengths = torch.LongTensor(word_lengths)

        batch = torch.LongTensor(rs)  # B x num_f x seq_len
        targets = torch.LongTensor(targets)
        if cuda:
            yield (Variable(batch).cuda(),
                   Variable(targets).cuda(),
                   Variable(ps).cuda(),
                   Variable(cs).cuda(),
                   Variable(ps_lengths, requires_grad=False).cuda(),
                   Variable(cs_lengths, requires_grad=False).cuda(),
                   Variable(word_lengths, requires_grad=False).cuda())


        else:
            yield (Variable(batch),
                   Variable(targets),
                   Variable(ps),
                   Variable(cs),
                   Variable(ps_lengths, requires_grad=False),
                   Variable(cs_lengths, requires_grad=False),
                   Variable(word_lengths, requires_grad=False))



def train(args):

    if os.path.exists(args.save_dir):
        print("Make a new save directory.")
        sys.exit(0)
    else:
        os.mkdir(args.save_dir)

    # Fetch processed data. #### Data is preprocessed offline ####
    dataset, dicts = load_data(args.data_dir)
    dist1id = dicts["relpos"][0][1]
    ent0id = dicts["relpos"][0][0]

    if args.dataset == "SE17":
        labvocab = ["None", "Synonym-of", "Hyponym-of", "r_Hyponym-of"]
    else:
        labvocab = ["None", "COMPARE", "MODEL-FEATURE", "PART_WHOLE", "RESULT", "TOPIC",
                    "USAGE", "r_MODEL-FEATURE", "r_PART_WHOLE", "r_RESULT", "r_TOPIC", "r_USAGE"]

    labvocab = {n: i for (i, n) in enumerate(labvocab)}  # This includes None
    for dtype in dataset:
        dataset[dtype] = [([[dicts["word"][0].get(i, 0) for i in r[0]],
                            [dicts["relpos"][0].get(i, 0) for i in r[1]],
                            [dicts["relpos"][0].get(i, 0) for i in r[2]],
                            [dicts["ner"][0].get(i, 0) for i in r[3]],
                            [dicts["pos"][0].get(i, 0) for i in r[4]]],
                           labvocab.get(l))
                          for r, l in dataset[dtype]]

    # Split the training set into tr/dv, and use dev as the test set.
    # (Discard untouched test set)
    split_idx = int(len(dataset["train"]) * 0.75)
    dataset["test"] = dataset["dev"]
    np.random.shuffle(dataset["train"])
    dataset["dev"] = dataset["train"][split_idx:]
    dataset["train"] = dataset["train"][:split_idx]

    dataset["dev"] = upsample(dataset["dev"], args.upsample)
    dataset["train"] = upsample(dataset["train"], args.upsample)

    wvocab = dicts["word"][0]
    rlpvocab = dicts["relpos"][0]
    posvocab = dicts["pos"][0]
    entvocab = dicts["ner"][0]

    print("Tr: {}, Dv: {}, Ts: {}".format(len(dataset["train"]),
                                          len(dataset["dev"]),
                                          len(dataset["test"])))

    if args.embedding_file:
        W, embed_dim = load_pretrained_embedding(dicts["word"][0], args.embedding_file)
        assert embed_dim == args.embed_dim
    else:
        W = None

    if args.model == 'CNN':
        model = CNNclassifier(vocab_size=len(wvocab),
                              embed_dim=100,
                              out_dim=200,
                              n_ent_labels=len(entvocab),
                              n_pos_labels=len(posvocab),
                              n_loc_labels=len(rlpvocab),
                              n_rel_labels=len(labvocab),
                              pretrained_emb=W,
                              freeze_emb=args.freeze_emb)
    else:
        model = LSTMRelationClassifier(vocab_size=len(wvocab),
                              embed_dim=100,
                              hidden_dim=200,
                              n_ent_labels=len(entvocab),
                              n_pos_labels=len(posvocab),
                              n_loc_labels=len(rlpvocab),
                              n_rel_labels=len(labvocab),
                              pretrained_emb=W,
                              freeze_emb=args.freeze_emb)

    if args.cuda:
        model = model.cuda()

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=args.lr)

    n_batches = len(dataset["train"]) // args.batch_size + 1
    dev_n_batches = len(dataset["dev"]) // args.batch_size + 1

    #with open(os.path.join(args.save_dir, "train.log"), "a") as f:
    #    print("\t\t".join(["Epoch", "Tr loss", "Dv loss", "Macro-F1"]), file=f)

    macro_F1_best = 0    # For tracking best macro F1
    F1_best = None       # For per-class F1
    tolerance_count = 0  # For tracking the number of waits
    aggvocab = None

    for epoch in tqdm.trange(args.num_epochs, ncols=100, desc="Epoch"):
        epoch_loss = 0
        train_batches = batch_loader(dataset["train"], dist1id, ent0id,
                                     batch_size=args.batch_size,
                                     cuda=args.cuda)
        dev_batches = batch_loader(dataset["dev"], dist1id, ent0id,
                                   batch_size=args.batch_size,
                                   cuda=args.cuda)
        # Train
        model.train()
        for rs, targets, ps, cs, ps_lengths, cs_lengths, word_lengths in tqdm.tqdm(train_batches, total=n_batches,
                                     ncols=100, desc="Training"):

            optimizer.zero_grad()
            predictions, (word_predict_f, word_predict_b) = model(rs[:, 0, :].squeeze(1),
                                rs[:, 1, :].squeeze(1),
                                rs[:, 2, :].squeeze(1),
                                rs[:, 3, :].squeeze(1),
                                rs[:, 4, :].squeeze(1),
                                ps, cs, ps_lengths,
                                cs_lengths, word_lengths)

            # Create word targets, and have -100 for masked locations.
            words = rs[:, 0, :].squeeze(1)
            word_targets_f = words.squeeze(1).clone().data.fill_(-100)
            word_targets_b = words.squeeze(1).clone().data.fill_(-100)
            for i, (sent, word_length) in enumerate(zip(words, word_lengths)):
                idx = int(word_length.data.cpu().numpy()[0])
                word_targets_f[i][: idx-1] = sent.data[1: idx]
                word_targets_b[i][1 : idx] = sent.data[: idx-1]

            word_targets_flatten_f = word_targets_f.contiguous().view(-1)
            word_targets_flatten_b = word_targets_b.contiguous().view(-1)

            word_targets_flatten_f = Variable(word_targets_flatten_f).cuda()
            word_targets_flatten_b = Variable(word_targets_flatten_b).cuda()

            lamda = args.lamda
            loss = loss_function(predictions, targets) + lamda * (loss_function(word_predict_f, word_targets_flatten_f) + loss_function(word_predict_b, word_targets_flatten_b))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.data[0]

        # Devp
        # Compute F-1.
        TP = [0 for _ in range(len(labvocab))]
        FP = [0 for _ in range(len(labvocab))]
        FN = [0 for _ in range(len(labvocab))]

        model.eval()
        eval_loss = 0
        mistakes = []
        for rs, targets, ps, cs, ps_lengths, cs_lengths, word_lengths  in tqdm.tqdm(dev_batches, total=dev_n_batches,
                                     ncols=100, desc="Evaluating"):

            predictions, _ = model(rs[:, 0, :].squeeze(1),
                                rs[:, 1, :].squeeze(1),
                                rs[:, 2, :].squeeze(1),
                                rs[:, 3, :].squeeze(1),
                                rs[:, 4, :].squeeze(1),
                                ps, cs, ps_lengths,
                                cs_lengths, word_lengths)

            loss = loss_function(predictions, targets)
            eval_loss += loss.data[0]
            preds = predictions.data.max(dim=1)[1]

            for pred, r, y in zip(preds, rs.data, targets.data):
                if pred == y:
                    TP[y] += 1
                else:
                    FP[pred] += 1
                    FN[y] += 1
                    mistakes.append((denum(dicts, r), pred, y))

        # Aggregate artificially-created reverse relation type
        ATP, AFP, AFN = [], [], []
        ATP, aggvocab = aggregate_stats(labvocab, TP)
        AFP, _ = aggregate_stats(labvocab, FP)
        AFN, _ = aggregate_stats(labvocab, FN)

        P = [float(tp)/(tp+fp) if tp+fp > 0 else 0 for tp, fp in zip(ATP, AFP)]
        R = [float(tp)/(tp+fn) if tp+fn > 0 else 0 for tp, fn in zip(ATP, AFN)]
        F1 = [2*p*r/(p+r) if p+r > 0 else 0 for p, r in zip(P, R)]
        macro_F1 = np.mean(F1[1:])

        tqdm.tqdm.write(("Epoch {:2d}\ttr_loss: {:.3f}"
                         "\tdv_loss / F1: ({:.3f} | {:.3f})").format(
                             epoch+1, epoch_loss, eval_loss, macro_F1))

       # with open(os.path.join(args.save_dir, "train.log"), "a") as f:
       #     print("{:2d}\t\t{:3.2f}\t\t{:.2f}\t\t{:.3f}".format(
       #         epoch+1, epoch_loss, eval_loss, macro_F1),
       #           file=f)

        if macro_F1 > macro_F1_best:
            save_checkpoint(model.state_dict(),
                            False,
                            os.path.join(args.save_dir, "model_best.pt"))
            macro_F1_best = macro_F1
            F1_best = F1
            tolerance_count = 0
            # Dump mistakes
            with open(os.path.join(args.save_dir, "dev_mistakes.pkl"), "wb") as f:
                pickle.dump(mistakes, f)

        elif tolerance_count > args.tolerate:
            break

        else:
            tolerance_count += 1

    print("Best Macro F1: {}".format(macro_F1_best))
    print(tabulate([[t, f] for t, f in zip(aggvocab, F1_best)],
                   headers=("Tag", "F1")))
    print()


    test_batches = batch_loader(dataset["test"], dist1id, ent0id,
                                batch_size=args.batch_size,
                                cuda=args.cuda)
    test_n_batches = len(dataset["test"]) // args.batch_size

    TP = [0 for _ in range(len(labvocab))]
    FP = [0 for _ in range(len(labvocab))]
    FN = [0 for _ in range(len(labvocab))]

    model.eval()
    eval_loss = 0

    for rs, targets, ps, cs, ps_lengths, cs_lengths, word_lengths  in tqdm.tqdm(test_batches, total=test_n_batches,
                                 ncols=100, desc="Evaluating"):

        predictions, _ = model(rs[:, 0, :].squeeze(1),
                            rs[:, 1, :].squeeze(1),
                            rs[:, 2, :].squeeze(1),
                            rs[:, 3, :].squeeze(1),
                            rs[:, 4, :].squeeze(1),
                            ps, cs, ps_lengths,
                            cs_lengths, word_lengths)

        loss = loss_function(predictions, targets)
        eval_loss += loss.data[0]
        preds = predictions.data.max(dim=1)[1]

        for pred, y in zip(preds, targets.data):
            if pred == y:
                TP[y] += 1
            else:
                FP[pred] += 1
                FN[y] += 1

    # Aggregate reverse artificially-created relation type
    ATP, AFP, AFN = [], [], []
    ATP, aggvocab = aggregate_stats(labvocab, TP)
    AFP, _ = aggregate_stats(labvocab, FP)
    AFN, _ = aggregate_stats(labvocab, FN)

    P = [float(tp)/(tp+fp) if tp+fp > 0 else 0 for tp, fp in zip(ATP, AFP)]
    R = [float(tp)/(tp+fn) if tp+fn > 0 else 0 for tp, fn in zip(ATP, AFN)]
    F1 = [2*p*r/(p+r) if p+r > 0 else 0 for p, r in zip(P, R)]
    macro_F1 = np.mean(F1[1:])

    print("Macro_F1: {}".format(macro_F1))
    print(tabulate([[t, p, r, f] for t, p, r, f in zip(aggvocab, P, R, F1)],
                   headers=("Tag", "P", "R", "F1")))
    with open(os.path.join(args.save_dir, "scores.txt"), "w") as f:
        print(tabulate([[t, p, r, f] for t, p, r, f in zip(aggvocab, P, R, F1)],
                       headers=("Tag", "P", "R", "F1")), file=f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Relation classifier")
    parser.add_argument("--model", type=str, default="CNN")
    parser.add_argument("--dataset", type=str, default="SE17")
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
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--lamda", type=float, default=1)
    parser.add_argument("--data-dir", type=str, default="data/scienceie")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Directory to save the experiment.")
    parser.add_argument("--upsample", type=float, help="Positive label duplication ratio.",
                        default=3)

    args = parser.parse_args()

    train(args)
