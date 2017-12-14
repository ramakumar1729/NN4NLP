import random
import re
import shutil

import torch
from torch.autograd import Variable
from torch.nn.init import xavier_uniform

from torch.nn.init import xavier_uniform

random.seed(1)

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
            cend = doc.index(list(filter(pat.match, doc[cbegin:]))[0])
            pbegin = doc.index("_P_")
            pend = doc.index(list(filter(pat.match, doc[pbegin:]))[0])

            # Design choice. Include P/C tag?
            xnew.append([doc[pbegin:pend], doc[cbegin:cend]])

        y = [i[0] for i in y]
    else:
        xnew = x

    none_idxs = [i for i, l in enumerate(y) if l == "NONE"]
    n_nones = len(none_idxs)
    # Randomly take away 90% of Nones
    remove_none_idxs = random.sample(none_idxs, k=9*(n_nones//10))
    xnew = [x_ for i, x_ in enumerate(xnew) if i not in remove_none_idxs]
    y = [y_ for i, y_ in enumerate(y) if i not in remove_none_idxs]

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

    # Shuffle
    zipped = list(zip(xs, ys))
    random.shuffle(zipped)
    xs, ys = zip(*zipped)

    for i in range(len(xs)//batch_size):

        p, p_lens = pad([[xdict[tok] for tok in j[0]]
                         for j in xs[batch_size*i: batch_size*(i+1)]],
                        pad_symbol=1)
        p = Variable(torch.LongTensor(p))
        p_lens = Variable(torch.LongTensor(p_lens), requires_grad=False)
        c, c_lens = pad([[xdict[tok] for tok in j[1]]
                         for j in xs[batch_size*i: batch_size*(i+1)]],
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


def pad_by_vector(chunks):
    """Fill the list of tensors with zero vectors.

    Parameters:
        chunks (List[Variable]): List of variable-length variables/tensors.
    Returns:
        Variable : Padded variable.
        List : List of original sequence lengths.
    """
    lengths = list(map(len, chunks))
    max_len = max(lengths)
    dim = chunks[0].size(1)

    padded = [torch.cat([c, Variable(torch.zeros(max_len-c.size(0), dim)).cuda()],
                        dim=0)
              if c.size(0) < max_len else c
              for c in chunks]

    # Expand batch dim and concatenate ignored chunk[0] and the padded.
    return torch.cat([t.unsqueeze(0) for t in padded], dim=0), lengths


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
    xavier_uniform(W)
    for w, i in dictionary.items():
        if w in vocab_embed:
            W[i] = vocab_embed[w]
            n += 1

    return W, embed_dim


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    """From ImageNet example of PyTorch official repository; save the model
        details to a file.
    """
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")
