"""
Train model
"""

import time
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import model_calendargnn
import model_calendargnnattn
import config
import data_loader
import data_manager
import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['calendargnn', 'calendargnnattn'], required=False, default='calendargnn',
                        help='Model to use')
    parser.add_argument('--label', choices=['gender', 'income', 'age'], required=False, default='gender',
                        help='User label for prediction')
    parser.add_argument('--num_epochs', type=int, required=False, default=10,
                        help='Num of epochs for training')
    parser.add_argument('--rand_seed', type=int, required=False, default=config.rand_seed,
                        help='Random seed')
    parser.add_argument('--hidden_dim', type=int, required=False, default=256,
                        help='Dimension of hidden states')
    parser.add_argument('--pattern_dim', type=int, required=False, default=128,
                        help='Dimension of patterns')
    parser.add_argument('--best_model_file', required=False, default=config.best_model_file,
                        help='File for saving best model during training')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use GPU for training')
    parser.add_argument('--top', type=int, required=False, default=0)
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    """ Parse arguments """
    print('===' * 30)
    print('Arguments:')
    _args = vars(args)
    for _arg, _arg_v in _args.items():
        print(f' - {_arg}: {_arg_v}')
    torch.manual_seed(args.rand_seed)
    np.random.seed(args.rand_seed)
    if torch.cuda.is_available() and not args.cuda:
        print(' *** CUDA is available. Consider adding "--cuda" argument. ***')
    elif args.cuda and not torch.cuda.is_available():
        print(' *** CUDA is not available. Argument "--cuda" ignored. ***')
    device = torch.device('cuda') if torch.cuda.is_available() and args.cuda else torch.device('cpu')
    task = 'cls' if args.label in ('gender', 'income') else 'reg'

    """ Load dataset """
    print('---' * 30)
    print('Dataset:')
    dataset = data_loader.load_dataset()
    train_loader, valid_loader, test_loader, v_embs_np, l_embs_np = \
        data_manager.prepare_dataset(dataset, lb=args.label, _top=args.top)
    (num_v, emb_dim_v) = v_embs_np.shape
    (num_l, emb_dim_l) = l_embs_np.shape
    assert (num_v == emb_dim_v and num_l == emb_dim_l)

    """ Initializations """
    model, train_epoch_func, evaluate_epoch_func = None, None, None
    if args.model == 'calendargnn':
        model = model_calendargnn.CalendarGNN(in_dims=[num_v, num_l], emb_dims=[emb_dim_v, emb_dim_l],
                                              hid_dims=[args.hidden_dim] * 5, pat_dims=[args.pattern_dim] * 4,
                                              out_dim={'gender': 2, 'income': 10, 'age': 1}[args.label],
                                              embs_v=torch.from_numpy(v_embs_np), embs_l=torch.from_numpy(l_embs_np),
                                              device=device).to(device)
        train_epoch_func = model_calendargnn.train_epoch
        evaluate_epoch_func = model_calendargnn.eval_epoch
    else:
        model = model_calendargnnattn.CalendarGNNAttn(in_dims=[num_v, num_l], emb_dims=[emb_dim_v, emb_dim_l],
                                                      hid_dims=[args.hidden_dim] * 5,
                                                      out_dim={'gender': 2, 'income': 10, 'age': 1}[args.label],
                                                      embs_v=torch.from_numpy(v_embs_np),
                                                      embs_l=torch.from_numpy(l_embs_np), device=device).to(device)
        train_epoch_func = model_calendargnn.train_epoch
        evaluate_epoch_func = model_calendargnn.eval_epoch
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=True)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(reduction='mean') if task == 'cls' else nn.MSELoss(reduction='mean')
    print('---' * 30)
    print('Model:')
    print(f' - Num of trainable parameters: {utils.count_num_params(model):,}')
    print(f' - Num of epochs: {args.num_epochs:,}')
    print(' - Optimizer: {}'.format(repr(optimizer).replace('\n', ';').replace('  ', ' ')))
    print(' - Device: {}'.format(f'{device} ({torch.get_num_threads()})' if device.type == 'cpu' else device))

    """ Training """
    print('---' * 30)
    print('Training:')
    best_epoch = (0, float('inf'))
    _sum_fmt_cls_bin = '{}: Loss={:7.3f}, Acc={:.2%}, AUC={:.4f}, F1={:.4f}, MCC={:.4f}; '
    # _sum_fmt_cls_mul_all = '{}: Loss={:7.3f}, Acc={:.2%}, F1-macro={:.4f}, F1-weighted={:.4f}, K={:.4f}; '
    _sum_fmt_cls_mul = '{}: Loss={:7.3f}, Acc={:.2%}, F1-macro={:.4f}, F1-micro={:.4f}, K={:.4f}; '
    _sum_fmt_reg = '{}: Loss={:7.3f}, R^2={:5.4}, MAE={:.4f}, RMSE={:.4f}, Pr={:.4f}, Sr={:.4f}; '
    for epoch in range(args.num_epochs):
        ''' Train and validate'''
        s_time = time.time()
        train_loss, train_metrics = train_epoch_func(model, train_loader, optimizer, args.label, criterion, device)
        valid_loss, valid_metrics = evaluate_epoch_func(model, valid_loader, args.label, criterion, device)
        e_time = time.time()
        ''' Track progress '''
        ep_mins, ep_secs = utils.sum_ep_time(s_time, e_time)
        if valid_loss < best_epoch[1]:
            best_epoch = (epoch, valid_loss)
            torch.save(model, args.best_model_file)
        ''' Print epoch summary '''
        print(f' - Epoch{epoch + 1:02d} ({ep_mins:2d}m{ep_secs:2d}s). ', end='')
        if args.label == 'gender':
            print(_sum_fmt_cls_bin.format('Train', train_loss, train_metrics[0], train_metrics[1],
                                          train_metrics[2], train_metrics[3]), end='')
            print(_sum_fmt_cls_bin.format('Valid', valid_loss, valid_metrics[0], valid_metrics[1],
                                          valid_metrics[2], valid_metrics[3]), flush=True)
        elif args.label == 'income':
            print(_sum_fmt_cls_mul.format('Train', train_loss, train_metrics[0], train_metrics[1],
                                          train_metrics[2], train_metrics[3]), end='')
            print(_sum_fmt_cls_mul.format('Valid', valid_loss, valid_metrics[0], valid_metrics[1],
                                          valid_metrics[2], valid_metrics[3]), flush=True)
        elif args.label == 'age':
            print(_sum_fmt_reg.format('Train', train_loss, train_metrics[0], train_metrics[1],
                                      train_metrics[2], train_metrics[3], train_metrics[4]), end='')
            print(_sum_fmt_reg.format('Valid', valid_loss, valid_metrics[0], valid_metrics[1],
                                      valid_metrics[2], valid_metrics[3], valid_metrics[4]), flush=True)

    """ Test and print summary """
    print('---' * 30)
    print('Summary:')
    print(f' - Best model at Epoch{best_epoch[0] + 1:02d}. ', end='')
    model = torch.load(args.best_model_file)
    test_loss, test_metrics = evaluate_epoch_func(model, test_loader, args.label, criterion, device)
    if args.label == 'gender':
        print(_sum_fmt_cls_bin.format('Test', 0, test_metrics[0], test_metrics[1],
                                      test_metrics[2], test_metrics[3]), flush=True)
    elif args.label == 'income':
        print(_sum_fmt_cls_mul.format('Test', 0, test_metrics[0], test_metrics[1],
                                      test_metrics[2], test_metrics[3]), flush=True)
    elif args.label == 'age':
        print(_sum_fmt_reg.format('Test', 0, test_metrics[0], test_metrics[1],
                                  test_metrics[2], test_metrics[3], test_metrics[4]), flush=True)
    print('===' * 30)
