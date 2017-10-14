import re

import torch
from torch.autograd import Variable


def load_data(data_path, mode="span"):

    # Entity end tag pattern
    pat = re.compile("_/entity.+")

    # Run preprocess.py before to combine x and y into one file with tab separation.
    with open(data_path, "r") as f:
        x, y = zip(*[line.strip().split("\t") for line in f])
        x = [s.split(" ") for s in x]
        y = [l.split(" ") for l in y]

    xnew = []
    if mode == "span":
        for doc in x:
            cbegin = doc.index("_C_")
            cend = doc.index(next(filter(pat.match, doc[cbegin:])))
            pbegin = doc.index("_P_")
            pend = doc.index(next(filter(pat.match, doc[pbegin:])))

            # Design choice. Include P/C tag?
            xnew.append([doc[pbegin:pend], doc[cbegin:cend]])

        y = [i[0] for i in y]
    else:
        xnew = x

    return xnew, y


def batch_loader(inputs, targets, batch_size, cuda=False):
    """Return a generator that iterates over data.

    Parameters:
        inputs (tuple(List, Dict)): A tuple of input data and the dictionary.
        targets (tuple(List, Dict)): A tuple of target data and the dictionary.
        batch_size (int): Batch size.
        cuda (bool): Whether to put tensors on GPU or not.
    Returns:
       generator of (input batch, target batch)
    """
    xs, xdict = inputs
    ys, ydict = targets
    for i in range(len(xs)//batch_size):

        p, p_lens = pad([[xdict[tok] for tok in i[0]]
                         for i in xs[batch_size*i: batch_size*(i+1)]],
                        pad_symbol=1)
        p = Variable(torch.LongTensor(p))
        p_lens = Variable(torch.LongTensor(p_lens), requires_grad=False)
        c, c_lens = pad([[xdict[tok] for tok in i[1]]
                         for i in xs[batch_size*i: batch_size*(i+1)]],
                        pad_symbol=1)
        c = Variable(torch.LongTensor(c))
        c_lens = Variable(torch.LongTensor(c_lens), requires_grad=False)
        y = torch.LongTensor([ydict[t]
                              for t in ys[batch_size*i: batch_size*(i+1)]])
        y = Variable(y)

        if cuda:
            p = p.cuda()
            p_lens = p_lens.cuda()
            c = c.cuda()
            c_lens = c_lens.cuda()
            y = y.cuda()

        yield p, p_lens, c, c_lens, y

def pad(data, pad_symbol=1):
    """Pad sequences with a padding index.
    Parameters:
        data List(int): Data to be padded. This can be either text-based,
            or numericalized form.
        pad_symbol int or str: Padding idx or padding symbol.
    """

    data = list(data)
    max_len = max(len(x) for x in data)
    lengths = [len(s) for s in data]

    padded = []
    for x in data:
        padded.append(
            list(x[:max_len]) +
            [pad_symbol] * max(0, max_len - len(x)))
    return padded, lengths

def load_pretrained_embedding(dictionary, embed_file, source="glove"):
    """Ref: https://github.com/Mjkim88/GA-Reader-Pytorch/blob/master/utils.py
    Parameters:
        dictionary (dict): A word-to-id dictionary.
        embed_file (str): Path the embeddings file.
        source (str): "glove" or "word2vec".
    Returns:
        torch.Tensor: Retrieved embedding matrix.
    """

    with open(embed_file, "r") as f:
        if source == "word2vec":
            # Word2vec has information at the first line
            info = f.readline().split()
            embed_dim = int(info[1])
        else:
            # Take embed_dim info from filename
            embed_dim = int(embed_file.split("/")[-1].split(".")[2][:-1])

        vocab_embed = {}
        for line in f:
            line = line.split()
            vocab_embed[line[0]] = torch.Tensor(list(map(float, line[1:])))

    vocab_size = len(dictionary)
    n = 0

    W = torch.randn((vocab_size, embed_dim))
    for w, i in dictionary.items():
        if w in vocab_embed:
            W[i] = vocab_embed[w]
            n += 1

    return W, embed_dim
