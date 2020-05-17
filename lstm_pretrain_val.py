import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import pickle

import os
import sys

from sklearn.preprocessing import StandardScaler

from torchdiffeq.torchdiffeq import odeint_adjoint

from fastai.text import *
from fastai.basic_train import Learner, LearnerCallback

from utils import *

def main(args):
    trn_files = os.listdir(args.input_path)

    items = []
    for fname in trn_files:
        if fname == '.DS_Store': continue
        df = pd.read_csv('%s/%s' % (args.input_path, fname))
        df = df.drop_duplicates('t')
        df = df.drop(['t'], axis=1)
        if len(df) < 2: continue
        items.append(df.values[::1000])
    items = np.stack(items, axis=0)
    n_samples, n_meas, n_vars = items.shape
    if args.model_path:
        with open(args.model_path + '_scaler.pkl', 'rb') as f:
            sc = pickle.load(f)
        items = sc.transform(items.reshape((-1, n_vars))).reshape((n_samples, n_meas, n_vars))
    else:
        sc = StandardScaler()
        items = sc.fit_transform(items.reshape((-1, n_vars))).reshape((n_samples, n_meas, n_vars))

    bs=args.bs
    len_train = int(0.5 * len(items))

    data_tx, data_vx, data_ty, data_vy = items[:len_train, :, :2],\
                                         items[len_train:, :, :2],\
                                         items[:len_train, :, 2:],\
                                         items[len_train:, :, 2:]
    
    x = ItemList(data_tx, label_cls=ItemList)._label_from_list(data_ty, from_item_lists=True)
    x.y.items = x.y.items.astype(np.float64)
    y = ItemList(data_vx, label_cls=ItemList)._label_from_list(data_vy, from_item_lists=True)
    y.y.items = y.y.items.astype(np.float64)
    bunch = GenLearnDataBunch.create(train_ds=x, valid_ds=y, bs=bs, bptt=30, path='lstm3')

    lrn = generator_model_learner(bunch, RAWD_LSTM, loss_func=torch.nn.MSELoss(), clip=0.5)

    if args.model_path:
        lrn = lrn.load(args.model_path)

    for epochs, lr in zip(args.epochs, args.lrs):
        lrn.fit_one_cycle(epochs, lr, moms=(0.7, 0.99), wd=1e-6)

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
    lrn.save_encoder(args.save_path + '_encoder')
    with open(args.save_path + '_scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_path', type=str, help='Input folder path')
    parser.add_argument('save_path', type=str, help='Output file path')
    parser.add_argument('--epochs', nargs='*', type=int, help='Epochs')
    parser.add_argument('--lrs', nargs='*', type=float, help='LRs')
    parser.add_argument('--data_size', type=int, default=150001, help='Dense data size')
    parser.add_argument('--bs', type=int, default=64, help='Batch size')
    parser.add_argument('--tmp_path', type=str, default=None, help='Set to true to only use cpu.')
    parser.add_argument('--model_path', type=str, default=None, help='Batch size')
    args = parser.parse_args()
    main(args)
