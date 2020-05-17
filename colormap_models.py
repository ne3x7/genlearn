import numpy as np
import pandas as pd
import tqdm
tqdm.monitor_interval = 0

from torchtext.data import Field
from awd_lstm import GeneratorModelData

import torch
import dill as pickle
from IPython import display as dp
from functools import partial
from torch import optim

FIELD = Field(sequential=False, use_vocab=False, tensor_type=torch.FloatTensor)

PATH = '/Users/nickstulov/Desktop/Work/GenLearn/colormap/'

em_sz = 2
nh = 100
nl = 3

# train models
for R in R_range:
	TRN_PATH = f'{PATH}data_R_{R}_trn/'
	VAL_PATH = f'{PATH}data_R_{R}_val/'

	md = GeneratorModelData.from_csv_files(PATH, FIELD, TRN_PATH, VAL_PATH, TST_PATH, bptt=30)

	opt_fn = partial(optim.Adam, betas=(0.7, 0.99))

	learner = md.get_model(opt_fn, em_sz, nh, nl, 
		dropouti=0.05, dropout=0.05, wdrop=0.1, dropouth=0.05)

	learner.clip=0.5
	learner.fit(1e-3, 10, wds=1e-6, cycle_len=1, cycle_mult=2)
	learner.clip = 0.3
	learner.fit(1e-3, 10, wds=1e-6, cycle_len=10)

	learner.save(f'{PATH}models/model_R_{R}.h5')