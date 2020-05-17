from sklearn.preprocessing import StandardScaler
import os
import pickle
import numpy as np
import pandas as pd

PATH = '/gpfs/gpfs0/y.maximov/kolya/data/rnd_params/mix'
items = []
for fname in os.listdir(PATH):
    if fname == '.DS_Store': continue
    df = pd.read_csv('%s/%s' % (PATH, fname))
    df = df.drop_duplicates('t')
    df = df.drop(['t'], axis=1)
    if len(df) < 2: continue
    items.append(df.values[::1000])
items = np.stack(items, axis=0)
n_samples, n_meas, n_vars = items.shape
sc = StandardScaler()
items = sc.fit_transform(items.reshape((-1, n_vars))).reshape((n_samples, n_meas, n_vars))
with open(PATH + '_scaler.pkl', 'wb') as f:
    pickle.dump(sc, f)
