"""
Load dataset
"""

import datetime
import collections

from config import *


def load_info_f(info_f, _del='\t'):
    ids = []
    with open(info_f, 'r') as f:
        next(f)
        for line in f:
            ts = line.strip().split(_del)
            assert len(ts) >= 2
            ids.append(ts[0])
    assert len(ids) == len(set(ids))  # ids are unique
    return ids


def load_user2sessions_f(user2sessions_f, _uids, _sids, _lids, _del1='\t', _del2=',', strict=True):
    _uid2_u = {_id: idx for idx, _id in enumerate(_uids)}
    _sid2_s = {_id: idx for idx, _id in enumerate(_sids)}
    _lid2_l = {_id: idx for idx, _id in enumerate(_lids)}
    ''' Parse user2sessions file '''
    _u2_sids, _sid2_lid = collections.defaultdict(list), {}
    with open(user2sessions_f, 'r') as f:
        for line_idx, line in enumerate(f):
            ts = line.strip().split(_del1)
            r_uid = ts[0]
            if r_uid in _uid2_u:  # if raw uid is known
                [r_sids, r_lids] = zip(*[t.split(_del2) for t in ts[1:]])
                for (r_sid, r_lid) in zip(r_sids, r_lids):
                    if r_sid in _sid2_s and r_lid in _lid2_l:  # if raw sid and raw lid are known
                        if r_sid not in _sid2_lid:  # if raw sid is unique
                            _u2_sids[_uid2_u[r_uid]].append(r_sid)
                            _sid2_lid[r_sid] = r_lid
    ''' Sort users by _uids '''
    [_us, u_sids] = zip(*sorted(list(_u2_sids.items()), key=lambda x: x[0]))
    assert all([len(_sids) >= 1 for _sids in u_sids])  # each user has > 1 sessions
    uids = [_uids[_u] for _u in _us]
    uid2u = {_id: idx for idx, _id in enumerate(uids)}
    if strict:
        # all users in _uids appear in user2sessions file
        assert all([uid == _uids[i] for i, uid in enumerate(uids)])
    else:
        # partial users in _uids appear in user2sessions file, and uids have same order with _uids
        all([uid in _uid2_u and _uid2_u[uid] >= i for i, uid in enumerate(uids)])
    ''' Sort sessions '''
    sids = [_sid for _sids in u_sids for _sid in _sids]
    assert len(sids) == len(set(sids))  # sids are unique
    sid2s = {_id: idx for idx, _id in enumerate(sids)}
    if strict:
        assert all([sid == _sids[i] for i, sid in enumerate(sids)])
    u2ss = [[sid2s[_sid] for _sid in _sids] for _sids in u_sids]
    ''' Sort locations '''
    s_lids = [_sid2_lid[sid] for sid in sids]
    lids = list(collections.OrderedDict.fromkeys(s_lids))
    lid2l = {_id: idx for idx, _id in enumerate(lids)}
    s2l = [lid2l[s_lid] for s_lid in s_lids]

    return uids, uid2u, sids, sid2s, lids, lid2l, u2ss, s2l


def _parse_dt_str(dt_str, _input_dt_str_fmt='%Y-%m-%dT%H:%M:%S', _output_dt_str_fmt='%j %H %W %w', _to_int=True):
    dt = datetime.datetime.strptime('2018-{}'.format(dt_str[:14]), _input_dt_str_fmt)  # skip tz %Z%z
    _dt_str = datetime.datetime.strftime(dt, _output_dt_str_fmt)
    dt_output = [int(t) for t in _dt_str.split()] if _to_int else _dt_str
    return dt_output


def load_session2items_f(session2items_f, sids, sid2s, _vids, _del1='\t', _del2=','):
    _vid2_v = {_id: idx for idx, _id in enumerate(_vids)}
    s2_vids, s_vid2dt_str = collections.defaultdict(list), {}
    s2vs, s_v2dt = [], {}
    with open(session2items_f, 'r') as f:
        for line_idx, line in enumerate(f):
            ts = line.strip().split(_del1)
            r_sid = ts[0]
            if r_sid in sid2s:  # if raw sid is known
                [r_vids, dt_strs] = zip(*[t.split(_del2) for t in ts[1:]])
                for (r_vid, dt_str) in zip(r_vids, dt_strs):
                    if r_vid in _vid2_v:  # if raw vid is known
                        s2_vids[sid2s[r_sid]].append(r_vid)
                        s_vid2dt_str[(sid2s[r_sid], r_vid)] = dt_str
    ''' Sort items '''
    s_vids = [vid for sid in sids for vid in s2_vids[sid2s[sid]]]
    vids = list(collections.OrderedDict.fromkeys(s_vids))
    vid2v = {_id: idx for idx, _id in enumerate(vids)}
    s2vs = [[vid2v[_vid] for _vid in s2_vids[sid2s[sid]]] for sid in sids]
    s_v2dt.update({(s, vid2v[_vid]): _parse_dt_str(dt_str) for (s, _vid), dt_str in s_vid2dt_str.items()})
    return vids, vid2v, s2vs, s_v2dt


def load_label_f(users_info_f, uids, uid2u, _del='\t'):
    _uid2lbs = collections.defaultdict(list)
    with open(users_info_f, 'r') as f:
        next(f)
        for line_idx, line in enumerate(f):
            ts = line.strip().split(_del)
            r_uid = ts[0]
            if r_uid in uid2u:  # if raw uid is known
                gender = 1 if ts[1] == 'M' else 0
                age = int(ts[2]) if ts[2] else 1
                income = int(ts[3]) if ts[3] else 0
                _uid2lbs[r_uid] = [gender, age, income]
    assert set(_uid2lbs.keys()) == set(uids)  # all users in uids must have labels
    ''' Sort labels '''
    u2lbs = [_uid2lbs[uid] for uid in uids]
    return u2lbs


def load_dataset():
    print(f'Loading {users_info_file}...')
    _uids = load_info_f(users_info_file)
    print(f'Loading {sessions_info_file}...')
    _sids = load_info_f(sessions_info_file)
    print(f'Loading {locations_info_file}...')
    _lids = load_info_f(locations_info_file)
    print(f'Loading {items_info_file}...')
    _vids = load_info_f(items_info_file)
    print(f'Loading {user2sessions_file}...')
    uids, uid2u, sids, sid2s, lids, lid2l, u2ss, s2l = load_user2sessions_f(user2sessions_file, _uids, _sids, _lids)
    print(f'Loading {session2items_file}...')
    vids, vid2v, s2vs, s_v2dt = load_session2items_f(session2items_file, sids, sid2s, _vids)
    print(f'Loading {users_info_file}...')
    u2lbs = load_label_f(users_info_file, uids, uid2u)
    dataset = {'uids': uids, 'uid2u': uid2u, 'sids': sids, 'sid2s': sid2s,
               'lids': lids, 'lid2l': lid2l, 'vids': vids, 'vid2v': vid2v,
               'u2ss': u2ss, 's2l': s2l, 's2vs': s2vs, 's_v2dt': s_v2dt,
               'u2lbs': u2lbs}
    return dataset
