

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


torch.manual_seed(1)

class LSTMRelationClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 n_ent_labels, n_pos_labels, n_rel_labels, n_loc_labels,
                 pretrained_emb=None, freeze_emb=False):
        """constructor of the relation classifier.

        Parameters:
            embeddings_dim (int): Dimension size of the look up table.
            hidden_dim (int): Dimension size of hidden reps in the RNN.
            vocab_size (int): Vocabulary size.
            n_rel_size (int): Number of target classes.
            pretrained_emb (torch.Tensor): Loaded tensor of word embeddings.
            freeze_emb (bool): The flag to freeze word embeddings weights.
        Returns:
            Variable: log softmax values for each class
        """
        super(LSTMRelationClassifier, self).__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_rel_labels = n_rel_labels

        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.location_embeddings = nn.Embedding(n_loc_labels, 10, padding_idx=1)
        self.pos_embeddings = nn.Embedding(n_pos_labels, 10, padding_idx=1)
        self.entity_embeddings = nn.Embedding(n_ent_labels, 10, padding_idx=1)

        if pretrained_emb is not None:
            self.word_embeddings.weight = nn.Parameter(pretrained_emb)

        # Whether or not to freeze the pretrained embeddings
        if freeze_emb:
            self.word_embeddings.weight.requires_grad = False

        self.dropout = nn.Dropout(0.5)

        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True,
                            batch_first=True)
        self.lstm2 = nn.LSTM(embed_dim+40, hidden_dim, bidirectional=True,
                            batch_first=True)

        self.fc = nn.Linear(2*hidden_dim, hidden_dim)
        # The linear layer that maps from hidden state space to tag space
        self.feat2tag = nn.Linear(3*hidden_dim, n_rel_labels)


        self.lm_fc_f  = nn.Linear(self.hidden_dim, self.vocab_size) 
        self.lm_fc_b  = nn.Linear(self.hidden_dim, self.vocab_size) 

    def forward(self, words, locs1, locs2, ents, poss, ps, cs, p_lengths, c_lengths, word_lengths):

        batch_size = len(words)

        # Parent entity embedding.

        p_emb = self.dropout(self.word_embeddings(ps))
        sorted_p_lens, sorted_p_idx = torch.sort(p_lengths, descending=True)
        _, unsort_p_idx = torch.sort(sorted_p_idx)
        packed_p_emb = pack(p_emb[sorted_p_idx],
                            lengths=sorted_p_lens.data.int().tolist(),
                            batch_first=True)
        # Last hidden state: 2 x B x hidden
        _, (p_hn, _) = self.lstm(packed_p_emb)
        # Concatenated last hidden state
        p_hn = p_hn.transpose(0, 1).contiguous().view(batch_size, -1)
        # Unsort
        p_hn = p_hn[unsort_p_idx]
        # Squeeze the dimension back to original hidden_dim
        p_hn = self.fc(p_hn)

        # Child entity embedding.
        
        c_emb = self.dropout(self.word_embeddings(cs))
        sorted_c_lens, sorted_c_idx = torch.sort(c_lengths, descending=True)
        _, unsort_c_idx = torch.sort(sorted_c_idx)
        packed_c_emb = pack(c_emb[sorted_c_idx],
                            lengths=sorted_c_lens.data.int().tolist(),
                            batch_first=True)
        # Last hidden state: 2 x B x hidden
        _, (c_hn, _) = self.lstm(packed_c_emb)
        # Concatenated last hidden state
        c_hn = c_hn.transpose(0, 1).contiguous().view(batch_size, -1)
        # Unsort
        c_hn = c_hn[unsort_c_idx]
        # Squeeze the dimension back to original hidden_dim
        c_hn = self.fc(c_hn)

        # Add sentence embedding.
        
        w_emb = self.dropout(self.word_embeddings(words))
        locs1_emb = self.dropout(self.location_embeddings(locs1))
        locs2_emb = self.dropout(self.location_embeddings(locs2))
        pos_emb = self.dropout(self.pos_embeddings(poss))
        ent_emb = self.dropout(self.entity_embeddings(ents))

        word_emb = torch.cat([w_emb, locs1_emb, locs2_emb, ent_emb, pos_emb], dim=2)
        # word_emb = w_emb
        sorted_word_lens, sorted_word_idx = torch.sort(word_lengths,
                descending=True)
        _, unsorted_word_idx = torch.sort(sorted_word_idx)
        packed_word_emb = pack(word_emb[sorted_word_idx],
                                lengths=sorted_word_lens.data.int().tolist(),
                                batch_first=True
                                )
        # sent_hn : 2x B x H.
        sent_outputs, (sent_hn, _) = self.lstm2(packed_word_emb)

        sent_outputs = unpack(sent_outputs)[0].transpose(0,1).contiguous()
        sent_outputs = sent_outputs[unsorted_word_idx]

        
        # Generate LM outputs.
        sent_outputs_f = sent_outputs[: , : , :self.hidden_dim] # B x S x H
        sent_outputs_b = sent_outputs[: , : , self.hidden_dim: ] # B x S x H

        sent_outputs_flatten_f = sent_outputs_f.contiguous().view(-1, self.hidden_dim) # B*S x H
        sent_outputs_flatten_b = sent_outputs_b.contiguous().view(-1, self.hidden_dim ) # B*S x H

        word_predict_flatten_f = self.lm_fc_f(sent_outputs_flatten_f)
        word_predict_flatten_b = self.lm_fc_b(sent_outputs_flatten_b)

        # word_predict_f = word_predict_flatten_f.view(batch_size, -1, self.vocab_size)
        # word_predict_b = word_predict_flatten_b.view(batch_size, -1, self.vocab_size)
        
        # sent_hn : B x 2 x H.
        sent_hn = sent_hn.transpose(0, 1).contiguous().view(batch_size, -1)
        sent_hn = sent_hn[unsorted_word_idx]
        sent_hn = self.fc(sent_hn)

        # features = p_hn + c_hn + sent_hn
        
        features = torch.cat((p_hn, c_hn, sent_hn), dim=1)
        scores = self.feat2tag(features)

        return scores, (word_predict_flatten_f, word_predict_flatten_b)



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

    def forward(self, words, locs1, locs2, ents, poss, ps, cs, p_lengths, c_lengths, word_lengths):
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
