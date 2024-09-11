import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset,DataLoader


class ExperienceBuffer(Dataset):
    #[reward,ea_reward,next_state,current_state,action,current_done,target_value,tmp_state_length,current_label]
    #[reward,ea_reward,next_state,current_state,action,current_done,target_value,tmp_state_length,target_env_value,env_action]
    def __init__(self,storage):
        rew_list = [a[0] for a in storage]
        env_rew_list = [a[1] for a in storage]
        act_list = [a[4] for a in storage]
        env_act_list = [a[9] for a in storage]#10
        s_list = [a[3] for a in storage]
        # sp_list = [a[1] for a in storage]
        q_list = [a[6] for a in storage]
        startPoints = [a[7] for a in storage]
        # trialLabel = [a[8] for a in storage]
        env_q_list = [a[8] for a in storage]#9
        self.env_rew = env_rew_list
        self.rew = rew_list
        self.start = startPoints
        # self.label = trialLabel
        self.act = act_list
        self.env_act = env_act_list
        self.s = s_list
        # self.sp = sp_list
        self.qval = q_list
        self.qval_env = env_q_list
    def __len__(self):
        return len(self.s)
    def __getitem__(self,idx):
        out ={}
        out['s'] = self.s[idx]
        out['rewards'] = self.rew[idx]
        out['env_rewards'] = self.env_rew[idx]
        # out['new_s'] = self.sp[idx]
        out['qval'] = self.qval[idx]
        out['env_qval'] = self.qval_env[idx]
        out['action'] = self.act[idx]
        out['env_action'] = self.env_act[idx]
        out['start'] = self.start[idx]
        # out['label'] = self.label[idx]
        return out
    def collate_fn(self,batch):
        data = list(batch)
        batch_s = torch.stack([torch.from_numpy(x['s']) for x in data]).float()
        #
        mask_s  = (batch_s != 0)#-10
        batch_rewards = torch.tensor(np.stack([x['rewards'] for x in data])).float()
        batch_env_rewards = torch.tensor(np.stack([x['env_rewards'] for x in data])).float()
        #
        batch_action = torch.tensor(np.stack([x['action'] for x in data])).long()
        batch_env_action = torch.tensor(np.stack([x['env_action'] for x in data])).long()
        batch_qval = torch.tensor(np.stack([x['qval'] for x in data])).float()
        batch_env_qval = torch.tensor(np.stack([x['env_qval'] for x in data])).float()
        batch_startPoints = torch.tensor(np.stack([x['start'] for x in data])).long()
        # batch_label = torch.tensor(np.stack([x['label'] for x in data])).long()
        return batch_s,batch_rewards,batch_action,batch_qval,mask_s,batch_startPoints,batch_env_rewards,batch_env_qval,batch_env_action


class OFC_Dataset_Ensemble_Time(Dataset):
    def __init__(self,data,value_data,sid,scaler=None):
        self.data = data
        self.valuedata = value_data
        self.scaler = scaler
        self.sid = sid[:,0]
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self,idx):
        out ={}
        if self.scaler != None:
          out['trialdata'] = torch.from_numpy(self.scaler.transform(self.data[idx,:,:].T).T) # originally (self.data[idx,:,:]
        else:
          out['trialdata'] = torch.from_numpy(self.data[idx,:,:])
        out['value'] = self.valuedata[idx]
        out['stim'] = self.sid[idx]
        return out
    def collate_fn(self,batch):
        data = list(batch)
        trials = torch.permute(torch.stack([x['trialdata'] for x in data]),(0,2,1)).float()
        # trials = trials  + torch.zeros(trials.shape).uniform_(-1,to=1)
        values = torch.tensor([x['value'] for x in data]).float()
        stimids = torch.tensor([x['stim'] for x in data]).long()
        return (trials,values,stimids)
