import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


torch.manual_seed(1)


class NRE(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,
                 pretrained_emb=None, freeze_emb=True):
        """constructor of the relation classifier.

        Parameters:
            embeddings_dim (int): Dimension size of the look up table.
            hidden_dim (int): Dimension size of hidden reps in the RNN.
            vocab_size (int): Vocabulary size.
            tagset_size (int): Number of target classes.
            pretrained_emb (torch.Tensor): Loaded tensor of word embeddings.
            freeze_emb (bool): The flag to freeze word embeddings weights.
        Returns:
            Variable: log softmax values for each class
        """
        super(NRE, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        if pretrained_emb is not None:
            self.word_embeddings.weight = nn.Parameter(pretrained_emb)

        # Whether or not to freeze the pretrained embeddings
        if freeze_emb:
            self.word_embeddings.requires_grad = False

        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True,
                            batch_first=True)

        self.fc = nn.Linear(2*hidden_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.feat2tag = nn.Linear(2*hidden_dim, tagset_size)

        # self.hidden = self.init_hidden()

    # def init_hidden(self):
    #     return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
    #             autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, ps, p_lengths, cs, c_lengths):

        batch_size = len(p_lengths)

        p_emb = self.dropout(self.word_embeddings(ps))
        c_emb = self.dropout(self.word_embeddings(cs))

        sorted_p_lens, sorted_p_idx = torch.sort(p_lengths, descending=True)
        _, unsort_p_idx = torch.sort(sorted_p_idx)
        packed_p_emb = pack(p_emb[sorted_p_idx],
                            lengths=sorted_p_lens.data.int().tolist(),
                            batch_first=True)

        sorted_c_lens, sorted_c_idx = torch.sort(c_lengths, descending=True)
        _, unsort_c_idx = torch.sort(sorted_c_idx)
        packed_c_emb = pack(c_emb[sorted_c_idx],
                            lengths=sorted_c_lens.data.int().tolist(),
                            batch_first=True)

        # Last hidden state: 2 x B x hidden
        _, (p_hn, _) = self.lstm(packed_p_emb)
        _, (c_hn, _) = self.lstm(packed_c_emb)

        # Concatenated last hidden state
        p_hn = p_hn.transpose(0, 1).contiguous().view(batch_size, -1)
        c_hn = c_hn.transpose(0, 1).contiguous().view(batch_size, -1)

        # Unsort
        p_hn = p_hn[unsort_p_idx]
        c_hn = c_hn[unsort_c_idx]

        # Squeeze the dimension back to original hidden_dim
        p_hn = self.fc(p_hn)
        c_hn = self.fc(c_hn)

        features = torch.cat((p_hn, c_hn), dim=1)
        # features = p_hn + c_hn
        scores = self.feat2tag(features)

        return scores
