"""
CalendarGNN
"""

import torch
import torch.nn as nn

import utils


class CalendarGNN(nn.Module):
    def __init__(self, in_dims, emb_dims, hid_dims, pat_dims, out_dim, embs_v, embs_l, device):
        super().__init__()
        ''' Parameters '''
        self.in_dim_v, self.in_dim_l = in_dims
        self.emb_dim_v, self.emb_dim_l = emb_dims
        self.hid_dim_sess, self.hid_dim_hemb, self.hid_dim_wemb, self.hid_dim_yemb, self.hid_dim_lemb = hid_dims
        self.hid_dim_hpat, self.hid_dim_wpat, self.hid_dim_ypat, self.hid_dim_lpat = pat_dims
        self.out_dim = out_dim
        self.device = device
        self.patterns = {'h', 'w', 'y', 'l'}
        self.pat2edim = {'h': self.hid_dim_hemb, 'w': self.hid_dim_wemb, 'y': self.hid_dim_yemb, 'l': self.hid_dim_lemb}
        self.pat2pdim = {'h': self.hid_dim_hpat, 'w': self.hid_dim_wpat, 'y': self.hid_dim_ypat, 'l': self.hid_dim_lpat}
        self._user_emb_dim = sum([self.pat2pdim[pat] for pat in self.patterns])
        ''' Architecture '''
        self.agg_layer_v = nn.Embedding(self.in_dim_v + 1, self.emb_dim_v, padding_idx=0)
        self.agg_layer_l = nn.Embedding(self.in_dim_l, self.emb_dim_l)
        self.agg_layers = nn.ModuleDict({
            # 1     item embs --> session embs
            'item2sess': nn.GRU(self.emb_dim_v, self.hid_dim_sess, batch_first=True),
            # 2.1   session embs --> hour embs
            'sess2hemb': nn.GRU(self.hid_dim_sess, self.hid_dim_hemb, batch_first=True),
            # 2.2   session embs --> week embs
            'sess2wemb': nn.GRU(self.hid_dim_sess, self.hid_dim_wemb, batch_first=True),
            # 2.3   session embs --> weekday embs
            'sess2yemb': nn.GRU(self.hid_dim_sess, self.hid_dim_yemb, batch_first=True),
            # 2.4   session embs --> location embs
            'sess2lemb': nn.GRU(self.hid_dim_sess + self.emb_dim_l, self.hid_dim_lemb, batch_first=True),
            # 3.1   hour embs --> hour pattern
            'hemb2hpat': nn.GRU(self.hid_dim_hemb, self.hid_dim_hpat, batch_first=True),
            # 3.2   week embs --> week pattern
            'wemb2wpat': nn.GRU(self.hid_dim_wemb, self.hid_dim_wpat, batch_first=True),
            # 3.3   weekday embs --> weekday pattern
            'yemb2ypat': nn.GRU(self.hid_dim_yemb, self.hid_dim_ypat, batch_first=True),
            # 3.4   location embs --> location pattern
            'lemb2lpat': nn.GRU(self.hid_dim_lemb, self.hid_dim_lpat, batch_first=True)
        })
        self.fc_layer = nn.Linear(self._user_emb_dim, self.out_dim)
        ''' Misc '''
        self._load_embs(embs_v, embs_l)
        self._init_params()

    def _load_embs(self, embs_v, embs_l):
        assert isinstance(embs_v, torch.Tensor) and isinstance(embs_l, torch.Tensor)
        assert embs_v.size() == (self.in_dim_v, self.emb_dim_v) and embs_l.size() == (self.in_dim_l, self.emb_dim_l)
        # Item embeddings: pad first embedding with 0s
        padded_embs_v = torch.cat([torch.zeros((1, self.emb_dim_v), dtype=torch.double, device=self.device),
                                   embs_v.to(self.device)], dim=0)
        self.agg_layer_v.weight.data.copy_(padded_embs_v)
        self.agg_layer_v.weight.requires_grad = False
        # Location embeddings
        self.agg_layer_l.weight.data.copy_(embs_l)
        self.agg_layer_l.weight.requires_grad = False

    def _init_params(self):
        for param in self.parameters():
            if param.requires_grad is True:
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:  # bias terms
                    nn.init.constant_(param, 0.)

    def agg_sess_embs(self, u_s_vs):
        assert u_s_vs.dim() == 2, u_s_vs.size()
        (num_s, max_len) = u_s_vs.size()
        ''' Feed to item embedding layer '''
        embedded = self.agg_layer_v(u_s_vs)
        assert embedded.size() == (num_s, max_len, self.emb_dim_v)
        ''' Pack padded seqs as input '''
        _lens = torch.sum(u_s_vs > 0, dim=1, keepdim=False)
        packed_input = nn.utils.rnn.pack_padded_sequence(embedded, _lens, batch_first=True, enforce_sorted=False)
        assert packed_input.data.size() == (torch.sum(_lens), self.emb_dim_v)
        assert packed_input.batch_sizes.size()[0] == max_len
        ''' Feed to aggregation layer item2sess '''
        packed_output, hid_h = self.agg_layers['item2sess'](packed_input)
        assert packed_output.data.size() == (torch.sum(_lens), self.hid_dim_sess)
        assert hid_h.size() == (1, num_s, self.hid_dim_sess)
        hid = hid_h.squeeze(0)
        ''' Unpack output seqs (if needed) '''
        # output, output_lens = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        # # output: [batch_size, max_seq_len, num_directions * hid_dim]
        # # output over padding tokens are zero tensors
        # assert output.size() == (num_s, max_len, self.hid_dim_sess)
        # assert all([torch.equal(output[:, v-1, :][i], hid_h.squeeze(0)[i]) for i, v in enumerate(output_lens)])
        return hid

    def agg_temp_pattern(self, sess_embs, u_s_ts, pat):
        assert sess_embs.size()[1] == self.hid_dim_sess
        num_s = sess_embs.size()[0]
        assert u_s_ts.size() == (num_s, 4), u_s_ts.size()
        assert pat in {'h', 'w', 'y'}
        """ Build temp (hour/week/weekday) unit embs """
        ''' Extract session temp units. Take unique and count. '''
        sess_ts = u_s_ts[:, {'h': 1, 'w': 2, 'y': 3}[pat]]
        unique_ts, _inverse_idxs, counts = torch.unique(sess_ts, sorted=True, return_inverse=True,
                                                        return_counts=True, dim=0)
        sess_idxs = torch.argsort(_inverse_idxs)
        assert unique_ts.size() == counts.size() and _inverse_idxs.size() == sess_idxs.size()
        # Fixed shape temp embs tensor: [num of temp units, max num of sess in each unit, sess emb dim]
        # Note: for temp units never appear in all sess, its embs are padded by zero vectors.
        temp_embs = torch.zeros(unique_ts.size()[0], torch.max(counts).item(), self.hid_dim_sess,
                                requires_grad=False, device=self.device)
        ''' Fill in temp embs tensor '''
        for i, sess_t in enumerate(unique_ts):
            _sess_sidx = torch.sum(counts[: i]).item()
            _sess_eidx = _sess_sidx + counts[i].item()
            for j, sess_idx in enumerate(sess_idxs[_sess_sidx: _sess_eidx]):
                temp_embs[i, j] = sess_embs[sess_idx]
        assert torch.sum(temp_embs).item() != 0
        ''' Pack padded temp embs as input '''
        packed_input = nn.utils.rnn.pack_padded_sequence(temp_embs, counts, batch_first=True, enforce_sorted=False)
        assert packed_input.data.size() == (torch.sum(counts), self.hid_dim_sess)
        assert packed_input.batch_sizes.size()[0] == torch.max(counts).item()
        ''' Feed to aggregation layer sess2hemb/sess2wemb/sess2yemb '''
        _func = {'h': 'sess2hemb', 'w': 'sess2wemb', 'y': 'sess2yemb'}[pat]
        packed_output, hid_h = self.agg_layers[_func](packed_input)
        assert packed_output.data.size() == (torch.sum(counts), self.pat2edim[pat])
        assert hid_h.size() == (1, unique_ts.size()[0], self.pat2edim[pat])
        temp_unit_embs = hid_h
        """ Build temp (hour/week/weekday) patterns """
        ''' Feed to aggregation layer hemb2hpat/wemb2wpat/yemb2ypat '''
        _func = {'h': 'hemb2hpat', 'w': 'wemb2wpat', 'y': 'yemb2ypat'}[pat]
        output, hid_h = self.agg_layers[_func](temp_unit_embs)
        assert hid_h.size() == (1, 1, self.pat2pdim[pat]), hid_h.size()
        temp_pat = hid_h.squeeze(0)
        return temp_pat

    def agg_spat_pattern(self, sess_embs, u_s_l):
        assert sess_embs.size()[1] == self.hid_dim_sess
        num_s = sess_embs.size()[0]
        assert u_s_l.size()[0] == num_s
        """ Build spat unit embs """
        ''' Extract session spat units. Take unique and count. '''
        unique_ls, _inverse_idxs, counts = torch.unique(u_s_l, sorted=True, return_inverse=True,
                                                        return_counts=True, dim=0)
        sess_idxs = torch.argsort(_inverse_idxs)
        assert unique_ls.size() == counts.size() and _inverse_idxs.size() == sess_idxs.size()
        # Fixed shape temp embs tensor: [num of temp units, max num of sess in each unit, sess emb dim]
        # Note: for temp units never appear in all sess, its embs are padded by zero vectors.
        spat_embs = torch.zeros(unique_ls.size()[0], torch.max(counts).item(), self.hid_dim_sess + self.emb_dim_l,
                                requires_grad=False, device=self.device)
        ''' Fill in spat embs tensor '''
        for i, sess_l in enumerate(unique_ls):
            _sess_sidx = torch.sum(counts[: i]).item()
            _sess_eidx = _sess_sidx + counts[i].item()
            for j, sess_idx in enumerate(sess_idxs[_sess_sidx: _sess_eidx]):
                spat_embs[i, j] = torch.cat((sess_embs[sess_idx], self.agg_layer_l(sess_l)))
        assert torch.sum(spat_embs).item() != 0
        ''' Pack padded temp embs as input '''
        packed_input = nn.utils.rnn.pack_padded_sequence(spat_embs, counts, batch_first=True, enforce_sorted=False)
        assert packed_input.data.size() == (torch.sum(counts), self.hid_dim_sess + self.emb_dim_l)
        assert packed_input.batch_sizes.size()[0] == torch.max(counts).item()
        ''' Feed to aggregation layer sess2lemb '''
        packed_output, hid_h = self.agg_layers['sess2lemb'](packed_input)
        assert packed_output.data.size() == (torch.sum(counts), self.pat2edim['l'])
        assert hid_h.size() == (1, unique_ls.size()[0], self.pat2edim['l'])
        spat_unit_embs = hid_h
        """ Build spat patterns """
        ''' Feed to aggregation layer lemb2lpat '''
        output, hid_h = self.agg_layers['lemb2lpat'](spat_unit_embs)
        assert hid_h.size() == (1, 1, self.pat2pdim['l']), hid_h.size()
        spat_pat = hid_h.squeeze(0)
        return spat_pat

    def forward(self, u_s_vs, u_s_ts, u_s_l):
        assert u_s_vs.dim() == 2 and u_s_ts.dim() == 2 and u_s_l.dim() == 1
        assert u_s_vs.size()[0] == u_s_ts.size()[0] and u_s_vs.size()[0] == u_s_l.size()[0]
        num_s = u_s_vs.size()[0]
        ''' Agg session embs '''
        sess_embs = self.agg_sess_embs(u_s_vs)
        assert sess_embs.size() == (num_s, self.hid_dim_sess)
        ''' Build patterns '''
        patterns = []
        # Temp patterns
        for pat in (self.patterns - {'l'}):
            temp_pat = self.agg_temp_pattern(sess_embs, u_s_ts, pat)
            assert temp_pat.shape == (1, self.pat2pdim[pat])
            patterns.append(temp_pat)
        # Spat pattern
        if 'l' in self.patterns:
            spat_pattern = self.agg_spat_pattern(sess_embs, u_s_l)
            assert spat_pattern.shape == (1, self.pat2pdim['l'])
            patterns.append(spat_pattern)
        ''' Concatenating patterns into holistic user emb '''
        user_emb = torch.cat(patterns, dim=1)
        assert user_emb.shape == (1, self._user_emb_dim)
        ''' Feed to last fully connected layer '''
        score = self.fc_layer(user_emb)
        assert score.size() == (1, self.out_dim)
        return score


def train_epoch(model, loader, optimizer, label, criterion, device):
    assert label in {'gender', 'income', 'age'}
    out_dim = {'gender': 2, 'income': 10, 'age': 1}[label]
    model.train()
    num_train, loss = 0, 0
    scores, targets = None, None
    for batch in loader:
        optimizer.zero_grad()
        ''' Push tensors to device '''
        batch = list(map(lambda x: x.to(device), batch))
        [u_s_vs, u_s_ts, u_s_l, u_lb] = batch
        ''' Forward, eval loss, and backpropagate '''
        score = model(u_s_vs.squeeze(0), u_s_ts.squeeze(0), u_s_l.squeeze(0))
        assert score.size() == (1, out_dim)
        u_lb = u_lb.float() if out_dim == 1 else u_lb
        _loss = criterion(torch.squeeze(score, dim=1), u_lb)  # Remove 2nd dim for regression task
        _loss.backward()
        optimizer.step()
        ''' Progress '''
        num_train += 1
        loss += _loss.item()
        scores = score if scores is None else torch.cat((scores, score))
        targets = u_lb if targets is None else torch.cat((targets, u_lb))
    return loss/num_train, utils.scores_to_metrics(scores, targets, label)


def eval_epoch(model, loader, label, criterion, device):
    assert label in {'gender', 'income', 'age'}
    out_dim = {'gender': 2, 'income': 10, 'age': 1}[label]
    model.eval()
    num_eval, loss = 0, 0
    scores, targets = None, None
    with torch.no_grad():
        for batch in loader:
            ''' Push tensors to device '''
            batch = list(map(lambda x: x.to(device), batch))
            [u_s_vs, u_s_ts, u_s_l, u_lb] = batch
            ''' Forward, eval loss, and backpropagate '''
            score = model(u_s_vs.squeeze(0), u_s_ts.squeeze(0), u_s_l.squeeze(0))
            assert score.size() == (1, out_dim)
            u_lb = u_lb.float() if out_dim == 1 else u_lb
            _loss = criterion(torch.squeeze(score, dim=1), u_lb)  # Remove 2nd dim for regression task
            ''' Progress '''
            num_eval += 1
            loss += _loss.item()
            scores = score if scores is None else torch.cat((scores, score))
            targets = u_lb if targets is None else torch.cat((targets, u_lb))
        return loss/num_eval, utils.scores_to_metrics(scores, targets, label)
