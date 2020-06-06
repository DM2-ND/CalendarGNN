"""
Manage dataset
"""

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data


def prepare_dataset(dataset, lb, _top=0):
    assert lb in {'gender', 'age', 'income'}
    ''' Load dataset file(s) '''
    uids, uid2u = dataset['uids'], dataset['uid2u']
    # sids, sid2s = dataset['sids'], dataset['sid2s']
    lids, lid2l = dataset['lids'], dataset['lid2l']
    vids, vid2v = dataset['vids'], dataset['vid2v']
    u2ss, s2l = dataset['u2ss'], dataset['s2l']
    s2vs, s_v2dt = dataset['s2vs'], dataset['s_v2dt']
    u2lbs = dataset['u2lbs']
    ''' Prepare labels '''
    assert all(isinstance(lb, int) for lbs in u2lbs for lb in lbs)
    [u2gender, u2age, u2income] = zip(*u2lbs)
    u2lb = {'gender': u2gender, 'age': u2age, 'income': u2income}[lb]
    ''' Item and location initial embeddings '''
    v_embs_np, l_embs_np = np.eye(len(vids), dtype=np.float), np.eye(len(lids), dtype=np.float)
    assert np.sum(v_embs_np) == len(vids) and np.sum(l_embs_np) == len(lids)
    ''' Split train/valid/test '''
    _ds_size = len(uids) if _top == 0 else _top
    _train_size, _valid_size = int(_ds_size * 0.8), int(_ds_size * 0.1)
    _test_size = _ds_size - _train_size - _valid_size
    _perm_ds_idxs = np.random.permutation(_ds_size)
    train_us = _perm_ds_idxs[: _train_size]
    valid_us = _perm_ds_idxs[_train_size: -_test_size]
    test_us = _perm_ds_idxs[-_test_size:]
    assert (not set(train_us).intersection(valid_us)) and (not set(train_us).intersection(test_us))
    print(f' - Train/valid/test: {len(train_us):,}/{len(valid_us):,}/{len(test_us):,}')
    ''' Pack loaders'''
    train_loader = _build_loader(train_us, u2ss, s2vs, u2lb, s_v2dt, s2l, shuffle=True)
    valid_loader = _build_loader(valid_us, u2ss, s2vs, u2lb, s_v2dt, s2l, shuffle=False)
    test_loader = _build_loader(test_us, u2ss, s2vs, u2lb, s_v2dt, s2l, shuffle=False)

    return train_loader, valid_loader, test_loader, v_embs_np, l_embs_np


def _build_loader(us, u2ss, s2vs, u2lb, s_v2dt, s2l, shuffle, max_s=200, max_v=200):
    assert all([(s, s2vs[s][0]) in s_v2dt for ss in [u2ss[u] for u in us] for s in ss])  # all session times are known
    assert all([s < len(s2l) for ss in [u2ss[u] for u in us] for s in ss])  # all session locations are known
    ''' Sessions of items '''
    # truncate each user's sessions of items into size [<= max_s, <= max_v]
    _u_s_vs = [[s2vs[s][:max_v] for s in ss[:max_s]] for ss in [u2ss[u] for u in us]]
    assert all([len(s_vs) <= max_s for s_vs in _u_s_vs]) and all([len(vs) <= max_v for s_vs in _u_s_vs for vs in s_vs])
    u_s_vs = [nn.utils.rnn.pad_sequence([torch.tensor([v + 1 for v in vs]) for vs in s_vs], batch_first=True)
              for s_vs in _u_s_vs]
    assert all([_ten.size()[0] <= max_s and _ten.size()[1] <= max_v for _ten in u_s_vs])
    ''' Temporal signals '''
    _u_s_ts = [[s_v2dt[(s, s2vs[s][0])] for s in ss[:max_s]] for ss in [u2ss[u] for u in us]]
    assert all([len(s_ts) <= max_s for s_ts in _u_s_ts]) and all([len(ts) == 4 for s_ts in _u_s_ts for ts in s_ts])
    u_s_ts = [torch.tensor(s_ts, dtype=torch.long) for s_ts in _u_s_ts]
    assert all([_ten.size()[0] <= max_s and _ten.size()[1] == 4 for _ten in u_s_ts])
    ''' Location signals '''
    _u_s_l = [[s2l[s] for s in ss[:max_s]] for ss in [u2ss[u] for u in us]]
    assert all([len(s_l) <= max_s for s_l in _u_s_l])
    u_s_l = [torch.tensor(s_l, dtype=torch.long) for s_l in _u_s_l]
    ''' User labels '''
    u_lb = [torch.tensor(u2lb[u], dtype=torch.long) for u in us]
    assert len(u_s_vs) == len(u_s_ts) and len(u_s_vs) == len(u_s_l) and len(u_s_vs) == len(u_lb)
    dataset = list(zip(u_s_vs, u_s_ts, u_s_l, u_lb))
    # noinspection PyTypeChecker
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle)
    return loader
