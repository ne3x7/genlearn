import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys

from torchdiffeq.torchdiffeq import odeint_adjoint

from sklearn.preprocessing import StandardScaler

from fastai.text import *
from fastai.basic_train import Learner, LearnerCallback

from utils import *

def main(args):
    PATH = args.input_path
    fnames = os.listdir(PATH)
    n_samples = len(fnames)

    x0, v0, c, (p0, q0) = get_initial_values(P0=1997.9999999936396,
                                             Q0=967.9249699065775,
                                             V0=1.0,
                                             angle0=0.494677176989154,
                                             theta0=np.array([0.003, 0.3, 1.81, 1.76, 3.5, 8.,
                                                              10., 10., 0.05, 0.02, 0.015, 1.,
                                                              1., 0.0001, 200]))

    data_size = args.data_size
    items = []
    for fname in fnames:
        if fname == '.DS_Store': continue
        df = pd.read_csv('%s/%s' % (PATH, fname))
        df = df.drop_duplicates('t')
        df = df.drop(['t'], axis=1)
        if len(df) < 2: continue
        items.append(df.values[:data_size])
    items = np.stack(items, axis=0)
    n_samples, n_meas, n_vars = items.shape
    if args.sparse:
        items = items[:, ::1000, :]
        data_size_sparse = 151

    perm = np.random.permutation(len(items))
    items = items[perm]

    n_samples, n_meas, n_vars = items.shape
    if args.model_path:
        with open(args.input_path + '_scaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        items = sc.transform(items.reshape((-1, n_vars))).reshape((n_samples, n_meas, n_vars))
    else:
        sc = StandardScaler()
        items = sc.fit_transform(items.reshape((-1, n_vars))).reshape((n_samples, n_meas, n_vars))

    bs=args.bs
    len_train = int(0.8 * len(items))

    data_tx, data_vx, data_ty, data_vy = items[:len_train, :, :2],\
                                             items[len_train:, :, :2],\
                                             items[:len_train, :, 2:],\
                                             items[len_train:, :, 2:]
    
    x = ItemList(data_tx, label_cls=ItemList)._label_from_list(data_ty, from_item_lists=True)
    x.y.items = x.y.items.astype(np.float64)
    y = ItemList(data_vx, label_cls=ItemList)._label_from_list(data_vy, from_item_lists=True)
    y.y.items = y.y.items.astype(np.float64)
    bunch = PhysGenLearnDataBunch.create(train_ds=x, valid_ds=y, bs=bs, path='node2')

    if args.sparse:
        t = torch.linspace(0, 1, steps=data_size_sparse)
    else:
        t = torch.linspace(0, 1, steps=data_size)
    
    initial_x = np.repeat(x0.reshape((1, -1)), bs, 0)

    rhs = RHS2CUDA().double().cuda()
    if args.interm:
        ode = ODE3(rhs, 2, t.cuda(), torch.from_numpy(initial_x).cuda()).double().cuda()
    else:
        ode = ODE4(rhs, 2, t.cuda(), torch.from_numpy(initial_x).cuda()).double().cuda()

    lrn = GenLearnLearner(bunch, ode, loss_func=nn.MSELoss(), metrics=[])

    if args.model_path:
        lrn = lrn.load(args.model_path)

    for epochs, lr in zip(args.epochs, args.lrs):
        lrn.fit_one_cycle(epochs, lr)

        if args.tmp_path:
            lrn.save(args.tmp_path)

        pred, true = lrn.get_preds()
        pred = pred.numpy().transpose(1, 0, 2)
        true = true.numpy().transpose(1, 0, 2)
        true = np.concatenate(true, 0)
        pred = np.concatenate(pred, 0)
        stacked = sc.inverse_transform(np.concatenate([true, pred], -1))
        pred_ = stacked[:, 2:]
        stacked = sc.inverse_transform(np.concatenate([pred, true], -1))
        true_ = stacked[:, 2:]
        err = [np.sqrt(np.sum((t - p) ** 2)) / np.sqrt(np.sum(t ** 2)) \
               for t, p in zip(true_[:len(true_)//151*151].reshape((len(true_)//151, 151, 2)), 
                               pred_[:len(true_)//151*151].reshape((len(true_)//151, 151, 2)))]

        print('\n'.join(['RMSE',
                         'mean %.4f' % np.mean(err).round(4),
                         'median %.4f' % np.median(err).round(4),
                         'max %.4f' % np.max(err).round(4),
                         'min %.4f' % np.min(err).round(4),
                         '95%% percentile %.4f' % np.percentile(err, 95).round(4)]))

    lrn.save(args.save_path)

    with open('results.pkl', 'wb') as f:
        pickle.dump((true, pred), f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input folder path')
    parser.add_argument('save_path', type=str, help='Output file path')
    parser.add_argument('--epochs', nargs='*', type=int, help='Epochs')
    parser.add_argument('--lrs', nargs='*', type=float, help='LRs')
    parser.add_argument('--data_size', type=int, default=150001, help='Dense data size')
    parser.add_argument('--sparse', type=bool, default=False, help='Use sparse data')
    parser.add_argument('--interm', type=bool, default=False, help='Use intermediate steps')
    parser.add_argument('--bs', type=int, default=10, help='Batch size')
    parser.add_argument('--model_path', type=str, default=None, help='Batch size')
    parser.add_argument('--tmp_path', type=str, default=None, help='Set to true to only use cpu.')
    args = parser.parse_args()
    main(args)
