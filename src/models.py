import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

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

  
class Embedding(nn.Module):

    def __init__(self, params, pretrain=None, wvocab=None):
        super().__init__()
        self.params = params

        self.word_emb = nn.Embedding(num_embeddings=params.word_vocab_size,
                                     embedding_dim=params.word_emb_dim,
                                     padding_idx=1)

        if pretrain:
            W, dim = load_pretrained_embedding(wvocab, pretrain)
            self.word_emb.weight = nn.Parameter(W)

        # Do we freeze the pretrained embeddings?
        if not self.params.train_word_emb:
            self.word_emb.requires_grad = False

class NRE(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(NRE, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(self.hidden[0].view(1, -1))
        tag_scores = F.log_softmax(tag_space)

        return tag_scores
