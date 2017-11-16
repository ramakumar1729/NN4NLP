

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNclassifier(nn.Module):

    """CNN-based classifier proposed in http://www.aclweb.org/anthology/S17-2171
    """

    def __init__(self, vocab_size, embed_dim, out_dim,
                 n_ent_labels, n_pos_labels, n_rel_labels, n_loc_labels,
                 pretrained_emb=None, freeze_emb=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.kernels = [5]

        self.word_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.loc_emb = nn.Embedding(n_loc_labels, 10, padding_idx=1) # TODO: Embedding size.
        self.ent_emb = nn.Embedding(n_ent_labels, 10, padding_idx=1) # TODO: Embedding size.
        self.pos_emb = nn.Embedding(n_pos_labels, 10, padding_idx=1) # TODO: Embedding size.

        if pretrained_emb is not None:
            self.word_emb.weight = nn.Parameter(pretrained_emb)

        # Whether or not to freeze the pretrained embeddings
        if freeze_emb:
            self.word_emb.weight.requires_grad = False

        self.feature_size = embed_dim + 10 + 10 + 10 + 10 # TODO: Replace numbers

        self.conv = nn.ModuleList([nn.Conv1d(self.feature_size, out_dim, i)
                                   for i in self.kernels])
        # self.conv = nn.ModuleList([nn.Conv2d(1, out_dim, (i, self.feature_size))
        #                            for i in self.kernels])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(out_dim, n_rel_labels)

    def forward(self, words, locs1, locs2, ents, poss):
        wemb = self.word_emb(words)
        lemb1 = self.loc_emb(locs1)
        lemb2 = self.loc_emb(locs2)
        eemb = self.ent_emb(ents)
        pemb = self.pos_emb(poss)

        # x : B x D x len
        x = torch.cat([wemb, lemb1, lemb2, eemb, pemb], dim=2).transpose(1, 2)
        # Each output : B x D' x len'
        conv_outs = [c(x) for c in self.conv]
        # Each output : B x D'
        max_pool_outs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                         for i in conv_outs]

        scores = self.fc(self.dropout(torch.cat(max_pool_outs, dim=1)))

        return scores
