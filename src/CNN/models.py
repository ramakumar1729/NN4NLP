import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from src.NRE.utils import pad_by_vector

torch.manual_seed(1)

class LSTMRelationClassifier(nn.Module):

    """LSTM classifier takes as input feature vectors, and compute the context
    representation with one LSTM, and another LSTM for specifically representing
    parent and child representations.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 n_ent_labels, n_pos_labels, n_rel_labels, n_loc_labels,
                 pretrained_emb=None, freeze_emb=False, relpos_ids=None):
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


        self.lm_fc_f = nn.Linear(self.hidden_dim, self.vocab_size)
        self.lm_fc_b = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, words, locs1, locs2, ents, poss, ps, cs, p_lengths,
                c_lengths, word_lengths, return_feature=False):

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

        word_emb = torch.cat([w_emb, locs1_emb, locs2_emb, ent_emb, pos_emb],
                             dim=2)
        # word_emb = w_emb
        sorted_word_lens, sorted_word_idx = torch.sort(word_lengths,
                                                       descending=True)
        _, unsorted_word_idx = torch.sort(sorted_word_idx)
        packed_word_emb = pack(word_emb[sorted_word_idx],
                               lengths=sorted_word_lens.data.int().tolist(),
                               batch_first=True)

        # sent_hn : 2x B x H.
        sent_outputs, (sent_hn, _) = self.lstm2(packed_word_emb)

        sent_outputs = unpack(sent_outputs)[0].transpose(0, 1).contiguous()
        sent_outputs = sent_outputs[unsorted_word_idx]  # !!!!!

        if return_feature:
            return sent_outputs[unsorted_word_idx], p_hn, c_hn

        # <==== Langauge modeling part ====>
        # Generate LM outputs.
        sent_outputs_f = sent_outputs[:, :, :self.hidden_dim] # B x S x H
        sent_outputs_b = sent_outputs[:, :, self.hidden_dim:] # B x S x H

        sent_outputs_flatten_f = (sent_outputs_f
                                  .contiguous()
                                  .view(-1, self.hidden_dim)) # B*S x H
        sent_outputs_flatten_b = (sent_outputs_b
                                  .contiguous()
                                  .view(-1, self.hidden_dim)) # B*S x H

        word_predict_flatten_f = self.lm_fc_f(sent_outputs_flatten_f)
        word_predict_flatten_b = self.lm_fc_b(sent_outputs_flatten_b)

        # sent_hn : B x 2 x H.
        sent_hn = sent_hn.transpose(0, 1).contiguous().view(batch_size, -1)
        sent_hn = sent_hn[unsorted_word_idx]
        sent_hn = self.fc(sent_hn)

        # features = p_hn + c_hn + sent_hn

        features = torch.cat((p_hn, c_hn, sent_hn), dim=1)
        scores = self.feat2tag(features)

        return scores, (word_predict_flatten_f, word_predict_flatten_b)


class LSTMRelationClassifierContext(nn.Module):

    """LSTM classifier takes as input feature vectors, and compute the context
    representation with one LSTM, and another LSTM for specifically representing
    parent and child representations.
    """

    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 n_ent_labels, n_pos_labels, n_rel_labels, n_loc_labels,
                 pretrained_emb=None, freeze_emb=False, relpos_ids=None):
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
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.n_rel_labels = n_rel_labels
        self.relpos_ids = relpos_ids
        self.aggregate = "last"

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
        self.feat2tag = nn.Linear(5*hidden_dim, n_rel_labels)

        self.lm_fc_f = nn.Linear(self.hidden_dim, self.vocab_size)
        self.lm_fc_b = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, words, locs1, locs2, ents, poss, ps, cs, p_lengths,
                c_lengths, word_lengths, return_feature=False):

        batch_size = len(words)

        # Parent entity embedding
        p_emb = self.dropout(self.word_embeddings(ps))
        sorted_p_lens, sorted_p_idx = torch.sort(p_lengths, descending=True)
        _, unsort_p_idx = torch.sort(sorted_p_idx)
        packed_p_emb = pack(p_emb[sorted_p_idx],
                            lengths=sorted_p_lens.data.int().tolist(),
                            batch_first=True)
        # Last hidden state: 2 x B x hidden
        _, (p_hn, _) = self.lstm(packed_p_emb)
        p_hn = p_hn.transpose(0, 1).contiguous().view(batch_size, -1)
        p_hn = p_hn[unsort_p_idx]
        p_hn = self.fc(p_hn)

        # Child entity embedding.
        c_emb = self.dropout(self.word_embeddings(cs))
        sorted_c_lens, sorted_c_idx = torch.sort(c_lengths, descending=True)
        _, unsort_c_idx = torch.sort(sorted_c_idx)
        packed_c_emb = pack(c_emb[sorted_c_idx],
                            lengths=sorted_c_lens.data.int().tolist(),
                            batch_first=True)
        _, (c_hn, _) = self.lstm(packed_c_emb)
        c_hn = c_hn.transpose(0, 1).contiguous().view(batch_size, -1)
        c_hn = c_hn[unsort_c_idx]
        c_hn = self.fc(c_hn)


        w_emb = self.dropout(self.word_embeddings(words))
        locs1_emb = self.dropout(self.location_embeddings(locs1))
        locs2_emb = self.dropout(self.location_embeddings(locs2))
        pos_emb = self.dropout(self.pos_embeddings(poss))
        ent_emb = self.dropout(self.entity_embeddings(ents))

        word_emb = torch.cat([w_emb, locs1_emb, locs2_emb, ent_emb, pos_emb],
                             dim=2)

        ent_spans = [slice(min(locs1[b].data.tolist().index(self.relpos_ids[0]),
                               len(locs2[b])-locs2[b].data.tolist()[::-1].index(self.relpos_ids[0])),
                           max(locs1[b].data.tolist().index(self.relpos_ids[0]),
                               len(locs2[b])-locs2[b].data.tolist()[::-1].index(self.relpos_ids[0])))
                     for b in range(len(locs1))]
        pad_vector = Variable(torch.zeros(1, self.embed_dim+40)).cuda()
        left_embs = [word_emb[i, :sl.start, :]
                     if sl.start != 0 else pad_vector
                     for i, sl in enumerate(ent_spans)]
        left_embs, l_lengths = pad_by_vector(left_embs)
        l_lengths = Variable(torch.LongTensor(l_lengths)).cuda()
        right_embs = [word_emb[i, sl.stop:, :]
                      if sl.stop < word_emb[i].size(0) else pad_vector
                      for i, sl in enumerate(ent_spans)]
        right_embs, r_lengths = pad_by_vector(right_embs)
        r_lengths = Variable(torch.LongTensor(r_lengths)).cuda()
        entspan_embs = [word_emb[i, sl.start:sl.stop, :]
                        for i, sl in enumerate(ent_spans)]
        entspan_embs, e_lengths = pad_by_vector(entspan_embs)
        e_lengths = Variable(torch.LongTensor(e_lengths)).cuda()

        sorted_l_lens, sorted_l_idx = torch.sort(l_lengths, descending=True)
        _, unsort_l_idx = torch.sort(sorted_l_idx)
        sorted_r_lens, sorted_r_idx = torch.sort(r_lengths, descending=True)
        _, unsort_r_idx = torch.sort(sorted_r_idx)
        sorted_e_lens, sorted_e_idx = torch.sort(e_lengths, descending=True)
        _, unsort_e_idx = torch.sort(sorted_e_idx)

        packed_l_emb = pack(left_embs[sorted_l_idx],
                            lengths=sorted_l_lens.data.long().tolist(),
                            batch_first=True)
        packed_r_emb = pack(right_embs[sorted_l_idx],
                            lengths=sorted_r_lens.data.long().tolist(),
                            batch_first=True)
        packed_e_emb = pack(entspan_embs[sorted_l_idx],
                            lengths=sorted_e_lens.data.long().tolist(),
                            batch_first=True)

        if self.aggregate == "last":
            _,(l_hn, _) = self.lstm2(packed_l_emb)
            _,(r_hn, _) = self.lstm2(packed_r_emb)
            _,(e_hn, _) = self.lstm2(packed_e_emb)
            l_hn = l_hn.transpose(0, 1).contiguous().view(batch_size, -1)
            l_hn = self.fc(l_hn[unsort_l_idx])
            r_hn = r_hn.transpose(0, 1).contiguous().view(batch_size, -1)
            r_hn = self.fc(r_hn[unsort_r_idx])
            e_hn = e_hn.transpose(0, 1).contiguous().view(batch_size, -1)
            e_hn = self.fc(e_hn[unsort_e_idx])

        elif self.aggregate == "max":
            l_outs,(_, _) = self.lstm2(packed_l_emb)
            r_outs,(_, _) = self.lstm2(packed_r_emb)
            e_outs,(_, _) = self.lstm2(packed_e_emb)
            l_outs, _ = unpack(l_outs, batch_first=True)
            r_outs, _ = unpack(r_outs, batch_first=True)
            e_outs, _ = unpack(e_outs, batch_first=True)
            l_hn = F.max_pool1d(l_outs.transpose(1, 2), 10).squeeze(2)
            r_hn = F.max_pool1d(r_outs.transpose(1, 2), 10).squeeze(2)
            e_hn = F.max_pool1d(e_outs.transpose(1, 2), 10).squeeze(2)

        # features = p_hn + c_hn + l_hn + r_hn + e_hn
        features = torch.cat((p_hn, c_hn, l_hn, r_hn, e_hn), dim=1)
        scores = self.feat2tag(features)

        return scores


class CNNclassifier(nn.Module):

    """CNN-based classifier proposed in http://www.aclweb.org/anthology/S17-2171
    """

    def __init__(self, vocab_size, embed_dim, out_dim,
                 n_ent_labels, n_pos_labels, n_rel_labels, n_loc_labels,
                 pretrained_emb=None, freeze_emb=False, relpos_ids=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.out_dim = out_dim
        self.kernels = [5]

        self.word_emb = nn.Embedding(vocab_size, embed_dim, padding_idx=1)
        self.loc_emb = nn.Embedding(n_loc_labels, 10, padding_idx=1)
        self.ent_emb = nn.Embedding(n_ent_labels, 10, padding_idx=1)
        self.pos_emb = nn.Embedding(n_pos_labels, 10, padding_idx=1)

        if pretrained_emb is not None:
            self.word_emb.weight = nn.Parameter(pretrained_emb)

        # Whether or not to freeze the pretrained embeddings
        if freeze_emb:
            self.word_emb.weight.requires_grad = False

        self.feature_size = embed_dim + 10 + 10 + 10 + 10 # Replace numbers

        self.conv = nn.ModuleList([nn.Conv1d(self.feature_size, out_dim, i)
                                   for i in self.kernels])

        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(self.kernels)*out_dim, n_rel_labels)

    def forward(self, words, locs1, locs2, ents, poss, ps, cs, p_lengths,
                c_lengths, word_lengths):
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


class StackLSTMCNN(nn.Module):

    """LSTM => CNN"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_ent_labels,
                 n_pos_labels, n_rel_labels, n_loc_labels, pretrained_emb=None,
                 freeze_emb=None, relpos_ids=None):
        super().__init__()
        self.relpos_ids = relpos_ids
        self.lstm_clf = LSTMRelationClassifier(vocab_size, embed_dim,
                                               hidden_dim, n_ent_labels,
                                               n_pos_labels, n_rel_labels,
                                               n_loc_labels,
                                               pretrained_emb=pretrained_emb,
                                               freeze_emb=freeze_emb)

        # CNN part
        self.feature_size = 2 * hidden_dim
        self.kernels = [5]

        self.conv = nn.ModuleList([nn.Conv1d(self.feature_size, hidden_dim, i)
                                   for i in self.kernels])

        self.dropout = nn.Dropout(0.5)
        self.cnn_fc = nn.Linear(len(self.kernels)*hidden_dim, n_rel_labels)

    def forward(self, words, locs1, locs2, ents, poss, ps, cs, p_lengths,
                c_lengths, word_lengths):
        inputs = (words, locs1, locs2, ents, poss, ps, cs, p_lengths, c_lengths,
                  word_lengths)

        ent_spans = [slice(min(locs1[b].data.tolist().index(self.relpos_ids[0]),
                               len(locs2[b])-locs2[b].data.tolist()[::-1].index(self.relpos_ids[0])),
                           max(locs1[b].data.tolist().index(self.relpos_ids[0]),
                               len(locs2[b])-locs2[b].data.tolist()[::-1].index(self.relpos_ids[0])))
                     for b in range(len(locs1))]

        # lstm_hiddens: B x len x 2*h
        lstm_hiddens, p_hn, c_hn = self.lstm_clf(*inputs, return_feature=True)

        # Variable length
        hidden_spans = [hs[s, :] for hs, s in zip(lstm_hiddens, ent_spans)]

        # B x len' x 2*h
        padded_hidden_spans, _ = pad_by_vector(hidden_spans)

        conv_outs = [c(padded_hidden_spans.transpose(1, 2)) for c in self.conv]

        # Each output : B x D'
        max_pool_outs = [F.max_pool1d(i, i.size(2)).squeeze(2)
                         for i in conv_outs]

        scores = self.cnn_fc(self.dropout(torch.cat(max_pool_outs, dim=1)))

        return scores


class VotingLSTMCNN(nn.Module):

    """LSTM => CNN"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, n_ent_labels,
                 n_pos_labels, n_rel_labels, n_loc_labels, pretrained_emb=None,
                 freeze_emb=None, relpos_ids=None):
        super().__init__()
        self.lstm_clf = LSTMRelationClassifier(vocab_size, embed_dim,
                                               hidden_dim, n_ent_labels,
                                               n_pos_labels, n_rel_labels,
                                               n_loc_labels,
                                               pretrained_emb=pretrained_emb,
                                               freeze_emb=freeze_emb)

        self.cnn_clf = CNNclassifier(vocab_size, embed_dim, hidden_dim,
                                     n_ent_labels, n_pos_labels, n_rel_labels,
                                     n_loc_labels, pretrained_emb=pretrained_emb,
                                     freeze_emb=freeze_emb)

    def forward(self, words, locs1, locs2, ents, poss, ps, cs, p_lengths,
                c_lengths, word_lengths):

        inputs = (words, locs1, locs2, ents, poss, ps, cs, p_lengths, c_lengths,
                  word_lengths)

        lstm_preds, (lm_f, lm_b) = self.lstm_clf(*inputs, return_feature=False)
        cnn_preds = self.cnn_clf(*inputs)

        # Linear interpolation -- we want prediction whenever either model says
        # yes
        scores = (F.softmax(lstm_preds) + F.softmax(cnn_preds))/2.0

        return scores, (lm_f, lm_b)
