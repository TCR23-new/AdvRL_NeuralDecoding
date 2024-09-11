from torch import nn
import torch
from preproc_functions import PositionalEncoding

class EnvAgent(nn.Module):
    def __init__(self,latentDim,tempTerm,numTimeSteps,inputDim,num_actions):
        super(EnvAgent,self).__init__()
        self.latentDim =latentDim
        self.T = numTimeSteps
        self.num_actions = num_actions
        self.inputDim = inputDim
        self.tempTerm = tempTerm
        self.second_path = Trf_Rep(latentDim,self.T,self.inputDim)
        self.q_head = nn.Sequential(
        nn.Linear(self.latentDim,self.num_actions)
        )
    def forward(self,x):
        out2 = self.second_path(x[:,:,:].permute(0,2,1))
        qvals = self.q_head(out2)
        return qvals

class Trf_Rep(nn.Module):
  def __init__(self,d_mod,timeDim,N,device=torch.device('cpu')):
    super(Trf_Rep,self).__init__()
    self.N = N
    self.d_mod = d_mod
    self.timeDim = timeDim
    self.device = device
    self.drpRate= 0.1
    self.dimff = d_mod
    self.nlayers = 2
    self.nheads = 1
    self.rnn = nn.Identity()
    self.encoder_layerT2 = nn.TransformerEncoderLayer(d_model=self.d_mod, nhead=self.nheads,dim_feedforward=self.dimff,dropout=self.drpRate,activation='gelu',batch_first=True)#0.2 relu
    self.transformer_encoderT2 = nn.TransformerEncoder(self.encoder_layerT2, num_layers=self.nlayers,enable_nested_tensor=False)#2
    self.posEnc = PositionalEncoding(d_model = self.d_mod,max_len=self.timeDim+1,dropout=0)#0.1
    self.act = nn.ReLU()
    self.lin_embedT =nn.Linear(self.N,self.d_mod)
    self.act_ = nn.ReLU()
    self.clf_head_time = nn.Parameter(torch.rand(1,1,self.d_mod),requires_grad = True)
    self.apply(self._init_weights)
  def _init_weights(self, module):
      if isinstance(module, nn.Linear):
          nn.init.xavier_uniform_(module.weight)
  def forward(self,x):
    input_x = self.lin_embedT(x)
    x_Tpath= self.act_(input_x)
    clf_headT = self.clf_head_time.repeat(x.shape[0],1,1)
    new_input_T = torch.cat((clf_headT,x_Tpath),dim=1)
    new_input_T_pe = self.posEnc(torch.permute(new_input_T,(1,0,2))).permute(1,0,2) # use this
    trf_out_T = self.transformer_encoderT2(new_input_T_pe)
    lin_out = trf_out_T[:,0,:]
    return lin_out


class SRNN_Agent(nn.Module):
    def __init__(self,latentDim,tempTerm,numTimeSteps,inputDim,num_actions):
        super(SRNN_Agent,self).__init__()
        self.latentDim =latentDim
        self.T = numTimeSteps
        self.num_actions = num_actions
        self.inputDim = inputDim
        self.tempTerm = tempTerm
        self.second_path = Trf_Rep(latentDim,self.T,self.inputDim)
        self.q_head = nn.Sequential(
        nn.Linear(self.latentDim,self.num_actions)
        )
    def forward(self,x):
        out2 = self.second_path(x[:,:,:].permute(0,2,1))
        qvals = self.q_head(out2)
        return qvals
