import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

import os
import sys

from torchdiffeq.torchdiffeq import odeint_adjoint

from fastai.text import *
from fastai.basic_train import Learner, LearnerCallback

def get_initial_values(P0, Q0, V0, angle0, theta0, Sn=2220, Vn=400, fn=60, Sb=100, Vb=400):
    """
    Initializes generator state vector `x` and network terminal variables vector `V`
    in machine base from given power flow values in system base and angle in radians.
    """
    
    # unpack parameters guess
    ws = 1
    KD = 0
    ra, x1d, xd, xq, H, T1d0, Kw, Tw, Tpss1, Tpss2, Tr, Tavr1, Tavr2, Te, K0 = theta0
    
    # define transforms
    
    # vf_MB = vf * Vb / Vn
    wb = 2 * np.pi * fn
    S_SBtoMB = Sb / Sn
    V_MBtoSB = Vn / Vb
    I_MBtoSB = Sn * Vb / (Sb * Vn)
    Z_MBtoSB = Sb * Vn ** 2 / (Sn * Vb ** 2)
    
    # initialize stator quantitites
    
    p0 = P0 / Sb
    q0 = Q0 / Sb
    Vt0 = V0 * np.exp(1j * angle0)
    S0 = p0 - 1j * q0
    I0 = S0 / Vt0.conjugate()
    vr0 = Vt0.real
    vi0 = Vt0.imag
    ir0 = -I0.real
    ii0 = -I0.imag
    
    # initialize DQ-quantities
    
    w0 = 1
    delta0 = np.angle(Vt0 + (ra + 1j * xq) * Z_MBtoSB * I0)
    
    Vdq0 = Vt0 * (1 / V_MBtoSB) * np.exp(1j * (-delta0 + np.pi/2))
    Idq0 = I0 * (1 / I_MBtoSB) * np.exp(1j * (-delta0 + np.pi/2))
    
    vd0 = Vdq0.real
    vq0 = Vdq0.imag
    id0 = Idq0.real
    iq0 = Idq0.imag
    
    # initialize order 3
    
    e1q0 = vq0 + ra * iq0 + x1d * id0
    
    # initialize AVR
    
    v = np.abs(Vt0)
    vref = v
    vs0 = 0
    vm0 = v
    vf0 = (e1q0 + (xd - x1d) * id0) * V_MBtoSB
    vr0 = K0 * (1 - Tavr1 / Tavr2) * (vref + vs0 - vm0)
    
    # initialize PSS
    
    v20 = 0
    
    # constants
    
    pm = (vq0 + ra * iq0) * iq0 + (vd0 + ra * id0) * id0
    vsmin, vsmax = -0.2, 0.2
    vfmin, vfmax = -6.4, 7
    
    # pack values
    
    x = np.array([delta0, w0, v20, vs0, vm0, vr0, vf0, e1q0])
    V = np.array([vd0, vq0, id0, iq0])
    c = np.array([pm, vsmin, vsmax, vfmin, vfmax, vref, vf0, vs0, wb, KD])
    
    return x, V, c, (p0, q0)

class RHSTrue(nn.Module):
    def __init__(self, c):
        super(RHSTrue, self).__init__()
        
        self.theta = torch.from_numpy(np.array([0.003, 0.3, 1.81, 1.76, 3.5, 8.,
                                                10., 10., 0.05, 0.02, 0.015, 1.,
                                                1., 0.0001, 200]))
        self.c = c
        
    def forward(self, t, x, v):
        vd = v[:, 0] * torch.sin(x[:, 0] - v[:, 1])
        vq = v[:, 0] * torch.cos(x[:, 0] - v[:, 1])

        id = (self.theta[3] * x[:, 7] - self.theta[3] * vq - self.theta[0] * vd) / (self.theta[1] * self.theta[3] + self.theta[0] ** 2)
        iq = (self.theta[0] * x[:, 7] - self.theta[0] * vq + self.theta[1] * vd) / (self.theta[1] * self.theta[3] + self.theta[0] ** 2)

        p = 22.2 * (vd * id + vq * iq)
        q = 22.2 * (vq * id - vd * iq)
        
        pe = (vq + self.theta[0] * iq) * iq + (vd + self.theta[0] * id) * id

        return torch.stack([
            self.c[8] * (x[:, 1] - 1),
            (self.c[0] - pe) / (2 * self.theta[4]),
            self.theta[6] * (self.c[0] - pe) / (2 * self.theta[4]) - x[:, 2] / self.theta[7],
            (self.theta[8] * (self.theta[6] * (self.c[0] - pe)
                              / (2 * self.theta[4]) - x[:, 2]
                              / self.theta[7]) + x[:, 2] - x[:, 3]) / self.theta[9],
            (v[:, 0] - x[:, 4]) / self.theta[10],
            (self.theta[14] * (1 - self.theta[11] / self.theta[12]) * (self.c[5] + x[:, 3] - x[:, 4]) - x[:, 5]) / self.theta[12],
            ((x[:, 5] + self.theta[14] * self.theta[11] * (self.c[5] + x[:, 3] - x[:, 4])
              / self.theta[12] + self.c[6]) * (1 + self.c[7] * (v[:, 0] / x[:, 4] - 1)) - x[:, 6]) / self.theta[13],
            (- x[:, 7] - (self.theta[2] - self.theta[1]) * id + x[:, 6]) / self.theta[5]
        ], dim=1)

class RHSCUDA(nn.Module):
    # ra, x1d, xd, xq, H, T1d0, Kw, Tw, Tpss1, Tpss2, Tr, Tavr1, Tavr2, K0
    # 5, 7, 8, ..., 12
    # 6 -> 5, 13 -> 6
    def __init__(self, c):
        super(RHSCUDA, self).__init__()
        
        self.theta1 = torch.tensor([0.1046, 0.3008, 0.1811, 0.1755])
        self.theta2 = nn.Parameter(torch.rand(3))
        self.s = torch.from_numpy(np.array([0.01, 1., 10., 10., 10.,
                                            100., 1000])).cuda()
        self.c = c
        
    def forward(self, t, x, v):
        vd = v[:, 0] * torch.sin(x[:, 0] - v[:, 1])
        vq = v[:, 0] * torch.cos(x[:, 0] - v[:, 1])

        id = (self.s[3] * self.theta1[3] * x[:, 7] - self.s[3] * self.theta1[3] * vq - self.s[0] * self.theta1[0] * vd) / (self.s[1] * self.theta1[1] * self.s[3] * self.theta1[3] + (self.s[0] * self.theta1[0]) ** 2)
        iq = (self.s[0] * self.theta1[0] * x[:, 7] - self.s[0] * self.theta1[0] * vq + self.s[1] * self.theta1[1] * vd) / (self.s[1] * self.theta1[1] * self.s[3] * self.theta1[3] + (self.s[0] * self.theta1[0]) ** 2)
        p = 22.2 * (vd * id + vq * iq)
        q = 22.2 * (vq * id - vd * iq)
        
        pe = (vq + self.s[0] * self.theta1[0] * iq) * iq + (vd + self.s[0] * self.theta1[0] * id) * id

        return torch.stack([
            self.c[8] * (x[:, 1] - 1),
            (self.c[0] - pe) / (2 * self.s[4] * self.theta2[0]),
            self.s[5] * self.theta2[1] * (self.c[0] - pe) / (2 * self.s[4] * self.theta2[0]) - x[:, 2] / 10,
            (0.05 * (self.s[5] * self.theta2[1] * (self.c[0] - pe)
                              / (2 * self.s[4] * self.theta2[0]) - x[:, 2]
                              / 10) + x[:, 2] - x[:, 3]) / 0.02,
            (v[:, 0] - x[:, 4]) / 0.015,
            (self.s[6] * self.theta2[2] * (1 - 1 / 1.) * (self.c[5] + x[:, 3] - x[:, 4]) - x[:, 5]) / 1.,
            ((x[:, 5] + self.s[6] * self.theta2[2] * 1 * (self.c[5] + x[:, 3] - x[:, 4])
              / 1. + self.c[6]) * (1 + self.c[7] * (v[:, 0] / x[:, 4] - 1)) - x[:, 6]) / 1e-4,
            (- x[:, 7] - (self.s[2] * self.theta1[2] - self.s[1] * self.theta1[1]) * id + x[:, 6]) / 8.
        ], dim=1)

class RHS2CUDA(nn.Module):
    def __init__(self):
        super(RHS2CUDA, self).__init__()
        
        self.lin1 = nn.Linear(10, 10)
        self.lin2 = nn.Linear(10, 8)
        
    def forward(self, t, x, v):
        y = torch.cat([x, v], dim=1)
        y = self.lin1(y).relu()

        return self.lin2(y)

def dropout_mask(x:Tensor, sz:Collection[int], p:float):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1-p).div_(1-p)

class RHS3CUDA(nn.Module):
    def __init__(self):
        super(RHS3CUDA, self).__init__()
        
        self.lin1 = nn.Linear(10, 50)
        self.lin2 = nn.Linear(50, 100)
        self.lin3 = nn.Linear(100, 8)
        
    def forward(self, t, x, v):
        y = torch.cat([x, v], dim=-1)
        y = self.lin1(y).relu()
        y = self.lin2(y).relu()
        return self.lin3(y)

class ODE(nn.Module):
    def __init__(self, func, dim, t, x0):
        super(ODE, self).__init__()

        self.rhs = func
        self.dim = dim
        self.t = t
        self.y0 = x0
    
    def forward(self, x):
        y_exog = x.transpose(0, 1) # (T, M, D)
        preds = odeint_adjoint(self.rhs, self.y0, y_exog, self.t, method='euler') # (T, M, D)
        if torch.isnan(preds).any():
            raise ValueError('Diverged!')
        pqs = []
        
        for delta, e1q, vt, phi in zip(preds[:, :, 0], preds[:, :, -1], y_exog[:, :, 0], y_exog[:, :, 1]):
            vd = vt * torch.sin(delta - phi)
            vq = vt * torch.cos(delta - phi)
            
            denom = self.rhs.s[1] * self.rhs.theta1[1] * self.rhs.s[3] * self.rhs.theta1[3] \
            + (self.rhs.s[0] * self.rhs.theta1[0]) ** 2

            id = (self.rhs.s[3] * self.rhs.theta1[3] * e1q - self.rhs.s[3] * self.rhs.theta1[3] * vq \
                  - self.rhs.s[0] * self.rhs.theta1[0] * vd) / denom
            iq = (self.rhs.s[0] * self.rhs.theta1[0] * e1q - self.rhs.s[0] * self.rhs.theta1[0] * vq \
                  + self.rhs.s[1] * self.rhs.theta1[1] * vd) / denom
            
            p = 22.2 * (vd * id + vq * iq) # (M, )
            q = 22.2 * (vq * id - vd * iq) # (M, )
            
            pqs.append(torch.stack([p, q], dim=1)) # (M, D)
            
        ans = torch.stack(pqs, dim=0).transpose(0, 1)
        return ans

class ODE2(nn.Module):
    def __init__(self, func, dim, t, x0):
        super(ODE2, self).__init__()

        self.rhs = func
        self.dim = dim
        self.t = t
        self.y0 = x0
    
    def forward(self, x):
        y_exog = x.transpose(0, 1) # (T, M, D)
        preds = odeint_adjoint(self.rhs, self.y0, y_exog, self.t,
                               method='euler', options={'step_size': 1e-4}) # (T, M, D)
        if torch.isnan(preds).any():
            raise ValueError('Diverged!')
        pqs = []
        
        for delta, e1q, vt, phi in zip(preds[:, :, 0], preds[:, :, -1], y_exog[:, :, 0], y_exog[:, :, 1]):
            vd = vt * torch.sin(delta - phi)
            vq = vt * torch.cos(delta - phi)
            
            denom = self.rhs.s[1] * self.rhs.theta[1] * self.rhs.s[3] * self.rhs.theta[3] \
            + (self.rhs.s[0] * self.rhs.theta[0]) ** 2

            id = (self.rhs.s[3] * self.rhs.theta[3] * e1q - self.rhs.s[3] * self.rhs.theta[3] * vq \
                  - self.rhs.s[0] * self.rhs.theta[0] * vd) / denom
            iq = (self.rhs.s[0] * self.rhs.theta[0] * e1q - self.rhs.s[0] * self.rhs.theta[0] * vq \
                  + self.rhs.s[1] * self.rhs.theta[1] * vd) / denom
            
            p = 22.2 * (vd * id + vq * iq) # (M, )
            q = 22.2 * (vq * id - vd * iq) # (M, )
            
            pqs.append(torch.stack([p, q], dim=1)) # (M, D)
            
        ans = torch.stack(pqs, dim=0).transpose(0, 1)
        return ans

class ODE3(nn.Module):
    def __init__(self, func, dim, t, x0):
        super(ODE3, self).__init__()

        self.rhs = func
        self.lin1 = nn.Linear(8, 10)
        self.lin2 = nn.Linear(10, 2)
        self.drop_h = nn.Dropout(0.3)
        self.drop_o = nn.Dropout(0.2)
        self.dim = dim
        self.t = t
        self.y0 = x0
    
    def forward(self, x):
        y_exog = x.transpose(0, 1)
        preds = odeint_adjoint(self.rhs, self.y0, y_exog, self.t,
                               method='euler', options={'step_size': 1e-4}).transpose(0, 1)
        preds = self.drop_h(preds)
        ans = self.lin1(preds).relu()
        ans = self.drop_o(ans)
        return self.lin2(ans)

class ODE4(nn.Module):
    def __init__(self, func, dim, t, x0):
        super(ODE4, self).__init__()

        self.rhs = func
        self.lin1 = nn.Linear(8, 10)
        self.lin2 = nn.Linear(10, 2)
        self.drop_h = nn.Dropout(0.3)
        self.drop_o = nn.Dropout(0.2)
        self.dim = dim
        self.t = t
        self.y0 = x0
    
    def forward(self, x):
        y_exog = x.transpose(0, 1)
        preds = odeint_adjoint(self.rhs, self.y0, y_exog, self.t,
                               method='euler').transpose(0, 1)
        preds = self.drop_h(preds)
        ans = self.lin1(preds).relu()
        ans = self.drop_o(ans)
        return self.lin2(ans)

class ParamsClampCallback(LearnerCallback):
    def __init__(self, learn:Learner, clip_min:float=None, clip_max:float=None):
        super().__init__(learn)
        self.clip_min = clip_min
        self.clip_max = clip_max
        
    def on_batch_begin(self, **kwargs):
        if self.clip_min is not None and self.clip_max is not None:
            for p in self.learn.model.parameters(): p.data.clamp_(self.clip_min, self.clip_max)

    def on_batch_end(self, **kwargs):
        if self.clip_min is not None and self.clip_max is not None:
            for p in self.learn.model.parameters(): p.data.clamp_(self.clip_min, self.clip_max)

class GenLearnLearner(Learner):
    def __init__(self, data:DataBunch, model:nn.Module, clip:float=None,
                 params_clip_min:float=None, params_clip_max:float=None, **learn_kwargs):
        super().__init__(data, model, **learn_kwargs)
        if clip: self.callback_fns.append(partial(GradientClipping, clip=clip))
        if params_clip_min is not None and params_clip_max is not None:
            cb = ParamsClampCallback(self, clip_min=params_clip_min, clip_max=params_clip_max)
            self.callbacks.append(cb)
