import numpy as np
import torch
from torch import nn
from torch import Tensor
import math


def Correct_per_vdiff(correctVect,vdiff):
  uvds = np.unique(vdiff)
  output = np.zeros((uvds.shape[0],))
  total = np.zeros((uvds.shape[0],))
  corrVect = np.array(correctVect)
  for ix,uvd in enumerate(uvds):
    idx = np.nonzero(vdiff == uvd)[0]
    tmp_cv = corrVect[idx]
    output[ix] = 100*(np.mean(tmp_cv))
    total[ix] = idx.shape[0]
  return uvds,output,total

def CorrectFlattening(data):
  for i in range(data.shape[0]):
    if i == 0:
      data2 = data[i,:,:]
    else:
      data2 = np.concatenate((data2,data[i,:,:]),axis=1)
  return data2

def Value_CS_ShiraData(stimIDs):
  val_cs = np.zeros((len(stimIDs),2))
  for i in range(val_cs.shape[0]):
    val_cs[i,0] = int(str(int(stimIDs[i,0]))[1])
    val_cs[i,1] = int(str(int(stimIDs[i,0]))[2])
  return val_cs

def StimID_Labeling(stimid1,stimid2=[]):
  if len(stimid2) != 0:
    all_stim = np.concatenate((stimid1,stimid2),axis=1)
  else:
    all_stim = stimid1
  uids = np.unique(all_stim,axis=0)
  vect = np.zeros_like(stimid1)
  for ix,uid in enumerate(uids):
    idx = np.nonzero(all_stim==uid)[0]
    vect[idx] = ix
  return vect

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        if d_model%2 != 0:
          d_model += 1
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # self.register_buffer('pe', pe)
    def forward(self,x):
        x = x + self.pe[:x.shape[0],:,:x.shape[2]] #self.pe[:x.size(0)] x.shape[2]
        return self.dropout(x)
