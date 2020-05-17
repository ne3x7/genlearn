import warnings
import pickle
import pandas as pd
from fastai.imports import *
from fastai.torch_imports import *
from fastai.rnn_reg import LockedDropout, WeightDrop
from fastai.core import set_grad_enabled, SingleModel, to_gpu
from fastai.nlp import flip_tensor
from fastai.learner import Learner
from fastai.lm_rnn import seq2seq_reg, repackage_var, LinearDecoder, SequentialRNN
from torchtext.data import Dataset, RawField

IS_TORCH_04 = LooseVersion(torch.__version__) >= LooseVersion('0.4')

def get_model(emb_sz, n_hid, n_layers, dropout=0.4, dropouth=0.3, dropouti=0.5, wdrop=0.5, qrnn=False, bias=False, bidir=False):
    """Returns a SequentialRNN model.
    A RNN_Encoder layer is instantiated using the parameters provided.
    This is followed by the creation of a LinearDecoder layer.
    Also by default (i.e. tie_weights = True), the embedding matrix used in the RNN_Encoder
    is used to  instantiate the weights for the LinearDecoder layer.
    The SequentialRNN layer is the native torch's Sequential wrapper that puts the RNN_Encoder and
    LinearDecoder layers sequentially in the model.
    Args:
        emb_sz (int): the embedding size to use to encode each token
        n_hid (int): number of hidden activation per LSTM layer
        n_layers (int): number of LSTM layers to use in the architecture
        dropouth (float): dropout to apply to the activations going from one LSTM layer to another
        dropouti (float): dropout to apply to the input layer.
        wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
        tie_weights (bool): decide if the weights of the embedding matrix in the RNN encoder should be tied to the
            weights of the LinearDecoder layer.
        qrnn (bool): decide if the model is composed of LSTMS (False) or QRNNs (True).
        bias (bool): decide if the decoder should have a bias layer or not.
    Returns:
        A SequentialRNN model
    """
    rnn_enc = RNN_Encoder(emb_sz, n_hid=n_hid, n_layers=n_layers, bidir=bidir,
                 dropouth=dropouth, dropouti=dropouti, wdrop=wdrop, qrnn=qrnn)
    return SequentialRNN(rnn_enc, LinearDecoder(emb_sz, emb_sz, dropout, tie_encoder=None, bias=bias))

class RNN_Learner(Learner):
    def __init__(self, data, models, **kwargs):
        super().__init__(data, models, **kwargs)

    def _get_crit(self, data): return F.mse_loss

    def save_encoder(self, name): save_model(self.model[0], self.get_model_path(name))

    def load_encoder(self, name): load_model(self.model[0], self.get_model_path(name))

class RNN_Encoder(nn.Module):

    """A custom RNN encoder network that uses
        - an embedding matrix to encode input,
        - a stack of LSTM or QRNN layers to drive the network, and
        - variational dropouts in the embedding and LSTM/QRNN layers
        The architecture for this network was inspired by the work done in
        "Regularizing and Optimizing LSTM Language Models".
        (https://arxiv.org/pdf/1708.02182.pdf)
    """

    initrange=0.1

    def __init__(self, emb_sz, n_hid, n_layers, bidir=False,
                 dropouth=0.3, dropouti=0.65, wdrop=0.5, qrnn=False):
        """ Default constructor for the RNN_Encoder class
            Args:
                bs (int): batch size of input data
                emb_sz (int): the embedding size to use to encode each token
                n_hid (int): number of hidden activation per LSTM layer
                n_layers (int): number of LSTM layers to use in the architecture
                dropouth (float): dropout to apply to the activations going from one LSTM layer to another
                dropouti (float): dropout to apply to the input layer.
                wdrop (float): dropout used for a LSTM's internal (or hidden) recurrent weights.
            Returns:
                None
          """

        super().__init__()
        self.ndir = 2 if bidir else 1
        self.bs, self.qrnn = 1, qrnn
        if self.qrnn:
            #Using QRNN requires cupy: https://github.com/cupy/cupy
            from .torchqrnn.qrnn import QRNNLayer
            self.rnns = [QRNNLayer(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(n_layers)]
            if wdrop:
                for rnn in self.rnns:
                    rnn.linear = WeightDrop(rnn.linear, wdrop, weights=['weight'])
        else:
            self.rnns = [nn.LSTM(emb_sz if l == 0 else n_hid, (n_hid if l != n_layers - 1 else emb_sz)//self.ndir,
                1, bidirectional=bidir) for l in range(n_layers)]
            if wdrop: self.rnns = [WeightDrop(rnn, wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)

        self.emb_sz,self.n_hid,self.n_layers = emb_sz,n_hid,n_layers
        self.dropouti = LockedDropout(dropouti)
        self.dropouths = nn.ModuleList([LockedDropout(dropouth) for l in range(n_layers)])

    def forward(self, input):
        """ Invoked during the forward propagation of the RNN_Encoder module.
        Args:
            input (Tensor): input of shape (sentence length x batch_size)
        Returns:
            raw_outputs (tuple(list (Tensor), list(Tensor)): list of tensors evaluated from each RNN layer without using
            dropouth, list of tensors evaluated from each RNN layer using dropouth,
        """
        sl, bs, nvars = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()
        with set_grad_enabled(self.training):
            raw_output = input
            new_hidden, raw_outputs, outputs = [],[],[]
            for l, (rnn, drop) in enumerate(zip(self.rnns, self.dropouths)):
                current_input = raw_output
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    raw_output, new_h = rnn(raw_output, self.hidden[l])
                new_hidden.append(new_h)
                raw_outputs.append(raw_output)
                if l != self.n_layers - 1: raw_output = drop(raw_output)
                outputs.append(raw_output)

            self.hidden = repackage_var(new_hidden)
        return raw_outputs, outputs

    def one_hidden(self, l):
        nh = (self.n_hid if l != self.n_layers - 1 else self.emb_sz)//self.ndir
        if IS_TORCH_04: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_())
        else: return Variable(self.weights.new(self.ndir, self.bs, nh).zero_(), volatile=not self.training)

    def reset(self):
        if self.qrnn: [r.reset() for r in self.rnns]
        self.weights = next(self.parameters()).data
        if self.qrnn: self.hidden = [self.one_hidden(l) for l in range(self.n_layers)]
        else: self.hidden = [(self.one_hidden(l), self.one_hidden(l)) for l in range(self.n_layers)]

class GeneratorModelDataLoader():
    def __init__(self, ds, bs, bptt, backwards=False):
        self.bs, self.bptt, self.backwards = bs, bptt, backwards
        data = sum([o.data for o in ds], []) # list of n_files numpy arrays
        fld = ds.fields['data']
        nums = fld.numericalize(data, device = None if torch.cuda.is_available() else -1)
        self.data = self.batchify(nums)
        self.i, self.iter = 0, 0
        self.n = len(self.data)

    def __iter__(self):
        self.i, self.iter = 0, 0
        return self

    def __len__(self): return self.n // self.bptt - 1

    def __next__(self):
        if self.i >= self.n-1 or self.iter>=len(self): raise StopIteration
        bptt = self.bptt if np.random.random() < 0.95 else self.bptt / 2.
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        res = self.get_batch(self.i, seq_len)
        self.i += seq_len
        self.iter += 1
        return res

    def batchify(self, data):
        nb = data.size(0) // self.bs
        data = data[:nb*self.bs]
        data = data.view(self.bs, -1, 4).transpose(0, 1).contiguous()
        if self.backwards: data=flip_tensor(data, 0)
        return to_gpu(data)

    def get_batch(self, i, seq_len):
        source = self.data
        seq_len = min(seq_len, len(source) - 1 - i)
        return source[i:i+seq_len, :, :2], source[i:i+seq_len, :, 2:]

class GeneratorModelDataset(Dataset):
    def __init__(self, path, field, **kwargs):
        with open('sc.pickle', 'rb') as f:
            sc = pickle.load(f)
        fields = [('data', field)]
        series = []
        if os.path.isdir(path):
            paths = glob(f'{path}/*.*')
        else:
            paths = [path]
        for p in paths:
            series.append(sc.transform(pd.read_csv(p).drop(['t'], axis=1).values))

        np.random.shuffle(series)

        examples = [torchtext.data.Example.fromlist([series], fields)]
        super().__init__(examples, fields, **kwargs)

class GeneratorModelData():
    def __init__(self, path, field, trn_ds, val_ds, test_ds, bs, bptt, backwards=False, **kwargs):
        """ Constructor for the class. An important thing that happens here is
            that the field's "build_vocab" method is invoked, which builds the vocabulary
            for this NLP model.
            Also, three instances of the LanguageModelLoader are constructed; one each
            for training data (self.trn_dl), validation data (self.val_dl), and the
            testing data (self.test_dl)
            Args:
                path (str): testing path
                field (Field): torchtext field object
                trn_ds (Dataset): training dataset
                val_ds (Dataset): validation dataset
                test_ds (Dataset): testing dataset
                bs (int): batch size
                bptt (int): back propagation through time
                kwargs: other arguments
        """
        self.bs = bs
        self.path = path
        self.trn_ds = trn_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

        factory = lambda ds: GeneratorModelDataLoader(ds, bs, bptt, backwards=backwards)
        self.trn_dl = factory(self.trn_ds)
        self.val_dl = factory(self.val_ds)
        self.test_dl = factory(self.test_ds)

    def get_model(self, opt_fn, emb_sz, n_hid, n_layers, **kwargs):
        """ Method returns a RNN_Learner object, that wraps an instance of the RNN_Encoder module.
        Args:
            opt_fn (Optimizer): the torch optimizer function to use
            emb_sz (int): embedding size
            n_hid (int): number of hidden inputs
            n_layers (int): number of hidden layers
            kwargs: other arguments
        Returns:
            An instance of the RNN_Learner class.
        """
        m = get_model(emb_sz, n_hid, n_layers, **kwargs)
        model = SingleModel(to_gpu(m))
        return RNN_Learner(self, model, opt_fn=opt_fn, crit=F.mse_loss)

    @classmethod
    def from_csv_files(cls, path, field, train, validation, test=None, bs=64, bptt=70, **kwargs):
        trn_ds, val_ds, test_ds = GeneratorModelDataset.splits(
            path, field=field, train=train, validation=validation, test=test)
        return cls(path, field, trn_ds, val_ds, test_ds, bs, bptt, **kwargs)