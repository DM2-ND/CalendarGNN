"""
CalendarGNN-Attn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CalendarGNNAttn(nn.Module):
    def __init__(self, in_dims, emb_dims, hid_dims, out_dim, embs_v, embs_l, device):
        super().__init__()
        ''' Parameters '''
        self.in_dim_v, self.in_dim_l = in_dims
        self.emb_dim_v, self.emb_dim_l = emb_dims
        self.hid_dim_sess, self.hid_dim_hemb, self.hid_dim_wemb, self.hid_dim_yemb, self.hid_dim_lemb = hid_dims
        self.out_dim = out_dim
        self.device = device
        self.pat2edim = {'h': self.hid_dim_hemb, 'w': self.hid_dim_wemb, 'y': self.hid_dim_yemb, 'l': self.hid_dim_lemb}
        self._user_emb_dim = sum([self.hid_dim_hemb, self.hid_dim_wemb, self.hid_dim_yemb,
                                  self.hid_dim_lemb, self.hid_dim_lemb, self.hid_dim_lemb])
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
        })
        self.attn_layers = nn.ModuleDict({
            # 3.1   location query emb + hour embs  --> hourly pattern impacted by locations
            'hpat_l': nn.Bilinear(self.hid_dim_lemb, self.hid_dim_hemb, 1),
            # 3.2   hour query emb + location embs  --> spatial pattern impacted by hours
            'lpat_h': nn.Bilinear(self.hid_dim_hemb, self.hid_dim_lemb, 1),
            # 3.3   location query emb + week embs  --> weekly pattern impacted by locations
            'wpat_l': nn.Bilinear(self.hid_dim_lemb, self.hid_dim_wemb, 1),
            # 3.4   week query emb + location embs  --> spatial pattern impacted by weeks
            'lpat_w': nn.Bilinear(self.hid_dim_wemb, self.hid_dim_lemb, 1),
            # 3.5   location query emb + weekday embs  --> weekday pattern impacted by locations
            'ypat_l': nn.Bilinear(self.hid_dim_lemb, self.hid_dim_yemb, 1),
            # 3.6   weekday query emb + location embs  --> spatial pattern impacted by weekdays
            'lpat_y': nn.Bilinear(self.hid_dim_yemb, self.hid_dim_lemb, 1),
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

    def agg_temp_unit_embs(self, sess_embs, u_s_ts, pat):
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
        temp_unit_embs = hid_h.squeeze(0)
        return temp_unit_embs

    def agg_spat_unit_embs(self, sess_embs, u_s_l):
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
        spat_unit_embs = hid_h.squeeze(0)
        return spat_unit_embs

    def attn_interactive_pattern(self, temp_unit_embs, pat, spat_unit_embs):
        assert temp_unit_embs.dim() == 2 and spat_unit_embs.dim() == 2 and pat in {'h', 'w', 'y'}
        assert temp_unit_embs.size()[1] == self.pat2edim[pat] and spat_unit_embs.size()[1] == self.hid_dim_lemb
        num_temp_units, num_spat_units = temp_unit_embs.size()[0], spat_unit_embs.size()[0]
        ''' Attend and generate temporal pattern impacted by spatial signals '''
        _spat_query = torch.mean(spat_unit_embs, dim=0, keepdim=True)
        assert _spat_query.size() == (1, self.hid_dim_lemb)
        _func = {'h': 'hpat_l', 'w': 'wpat_l', 'y': 'ypat_l'}[pat]
        _energys = self.attn_layers[_func](_spat_query.repeat(num_temp_units, 1), temp_unit_embs)
        assert _energys.size() == (num_temp_units, 1)
        tpat_l = torch.matmul(F.softmax(_energys, dim=0).view(1, -1), temp_unit_embs)
        assert tpat_l.size() == (1, self.pat2edim[pat])
        ''' Attend and generate spatial pattern impacted by temporal signals '''
        _temp_query = torch.mean(temp_unit_embs, dim=0, keepdim=True)
        assert _temp_query.size() == (1, self.pat2edim[pat])
        _func_re = {'h': 'lpat_h', 'w': 'lpat_w', 'y': 'lpat_y'}[pat]
        _energys_re = self.attn_layers[_func_re](_temp_query.repeat(num_spat_units, 1), spat_unit_embs)
        assert _energys_re.size() == (num_spat_units, 1)
        lpat_t = torch.matmul(F.softmax(_energys_re, dim=0).view(1, -1), spat_unit_embs)
        assert lpat_t.size() == (1, self.hid_dim_lemb)
        ''' Concatenate '''
        interactive_pattern = torch.cat((tpat_l, lpat_t), dim=1)
        assert interactive_pattern.size() == (1, self.pat2edim[pat] + self.hid_dim_lemb)
        return interactive_pattern

    def forward(self, u_s_vs, u_s_ts, u_s_l):
        assert u_s_vs.dim() == 2 and u_s_ts.dim() == 2 and u_s_l.dim() == 1
        assert u_s_vs.size()[0] == u_s_ts.size()[0] and u_s_vs.size()[0] == u_s_l.size()[0]
        num_s = u_s_vs.size()[0]
        ''' Agg session embs '''
        sess_embs = self.agg_sess_embs(u_s_vs)
        assert sess_embs.size() == (num_s, self.hid_dim_sess)
        ''' Build interactive patterns '''
        interactive_patterns = []
        for pat in ['h', 'w', 'y']:
            temp_unit_embs = self.agg_temp_unit_embs(sess_embs, u_s_ts, pat)
            spat_unit_embs = self.agg_spat_unit_embs(sess_embs, u_s_l)
            interactive_pattern = self.attn_interactive_pattern(temp_unit_embs, pat, spat_unit_embs)
            interactive_patterns.append(interactive_pattern)
        ''' Concatenating patterns into holistic user emb '''
        user_emb = torch.cat(interactive_patterns, dim=1)
        assert user_emb.shape == (1, self._user_emb_dim)
        ''' Feed to last fully connected layer '''
        score = self.fc_layer(user_emb)
        assert score.size() == (1, self.out_dim)
        return score
