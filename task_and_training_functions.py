import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import Dataset,DataLoader
from models import EnvAgent
from dataset_functions import ExperienceBuffer


class DataEnv_NVPAug2_3Actions_All_Adv():
    def __init__(self,data,val1,val2,choice,scaler,hidden_layer_size,otherParams,device,TotalSimExamples=200,ChoiceFlag = False,TrainFlag=True):
        self.scaler = scaler
        self.otherParams = otherParams
        self.total_examples = TotalSimExamples
        self.orig_data = data
        self.val1 = val1
        self.val2 = val2
        self.device = device
        self.vdiff = self.val1-self.val2
        self.TrainFlag = TrainFlag
        self.unique_vdiff = np.unique(self.vdiff)
        n_actions = np.unique(choice).shape[0]
        self.numClasses = np.unique(self.vdiff).shape[0]
        # self.EnvAgent  = EnvAgent0(n_actions = n_actions,numClasses=numClasses,hidden_size=hidden_layer_size)
        latentDim,tempTerm,timeDim,N = self.otherParams
        self.EnvAgent  = EnvAgent(latentDim,tempTerm,timeDim,N,self.numClasses)
        if ChoiceFlag == True:
          self.orig_choice = choice
        else:
          self.orig_choice = self.create_choice_vect()
        self.data = self.scale_data(self.orig_data[:,:,:])
        self.choice = self.orig_choice[:]
        self.current_state = 1
        self.current_Trial = 0
        self.current_Trial_count=0
        self.reward_ctr= 0
        self.env_reward_ctr = 0
        #
    def create_choice_vect(self):
        choice_vect = np.zeros((self.val1.shape[0],))
        for i in range(self.val1.shape[0]):
          if self.val1[i] > self.val2[i]:
            choice_vect[i] = 1
          elif self.val1[i] < self.val2[i]:
            choice_vect[i] = 0
          else:
            choice_vect[i] = np.random.choice([0,1])
        return choice_vect
    def data_length(self):
        return self.data.shape[0]
    def running_reward_ctr(self):
      return self.reward_ctr
    def scale_data(self,data):
      d = data.shape
      out = np.zeros((d[0],d[1],d[2]))
      for ii in range(data.shape[0]):
        for j in range(data.shape[2]):
          out[ii,:,j] = self.scaler.transform(data[ii,:,j].reshape(1,-1))[0,:]
      return out
    def next_state(self,action):
        end_of_trials = 0
        self.current_Trial_count += 1
        self.current_Trial += 1
        if self.current_Trial_count > self.data.shape[0]-1:
            print('current trial {} data shape {}'.format(self.current_Trial, self.data.shape[0]-1))
            self.current_Trial_count = 0
            self.current_Trial = 0
            end_of_trials = 1
        next_state = self.data[self.current_Trial,:,:]
        return next_state,end_of_trials
    # def next_state_Adv(self,action):
    #     end_of_trials = 0
    #     self.current_Trial +=1
    #     action1 = torch.tensor([action]).reshape(1,-1)
    #     out = self.EnvAgent(action1)
    #     env_action = torch.argmax(out,dim=1)
    #     ex_per_act =self.unique_vdiff[env_action]
    #     # find the examples that correspond to the action
    #     relevant_ex = np.nonzero(self.vdiff == ex_per_act)[0]
    #     ex_id = np.random.choice(relevant_ex)
    #     next_state = self.data[ex_id,:,:]
    #     if self.current_Trial > self.total_examples:
    #         end_of_trials = 1
    #     return next_state,end_of_trials,env_action
    def next_state_Adv(self,s):
        self.EnvAgent.eval()
        end_of_trials = 0
        # self.current_Trial +=1
        self.current_Trial_count +=1
        with torch.no_grad():
            out = self.EnvAgent(s.float().to(self.device))
        # env_action = torch.argmax(out,dim=1)
        env_action = np.random.choice(list(range(self.numClasses)),p=nn.functional.softmax(out[:,:],dim=1).reshape(-1).cpu().numpy())
        ex_per_act =self.unique_vdiff[env_action]
        # find the examples that correspond to the action
        relevant_ex = np.nonzero(self.vdiff == ex_per_act)[0]
        ex_id = np.random.choice(relevant_ex)
        self.current_Trial = ex_id
        next_state = self.data[ex_id,:,:]
        if self.current_Trial_count > self.total_examples:
            end_of_trials = 1
        self.EnvAgent.train()
        return next_state,end_of_trials,env_action
    def env_reset(self):
      # print('reseting the env')
      self.reward_ctr = 0
      self.env_reward_ctr=0
      self.current_Trial_count = 0
      self.current_state = 1
      self.data = self.scale_data(self.orig_data[:,:,:])
      self.choice = self.orig_choice[:]
      # return the current state
      if self.TrainFlag == True:
        s0_idx=np.random.choice(range(self.vdiff.shape[0]))
        self.current_Trial = s0_idx
      else:
        self.current_Trial = 0
      next_state = np.expand_dims(self.data[self.current_Trial,:,:],axis=0)
      return next_state
    def take_action(self,action):
        current_state = self.data[self.current_Trial,:,:]
        true_action = self.choice[self.current_Trial]
        reward=0
        env_agent_reward= 0
        if action == true_action:
            reward = 1
            # env_agent_reward = 0
        else:
            reward = 0
            # env_agent_reward += 1
        env_agent_reward = 1-reward
        #
        if self.TrainFlag == True:
            next_state,end_of_trials,env_action = self.next_state_Adv(torch.tensor(current_state).unsqueeze(0))
            # next_state = next_state.squeeze(0).numpy()
        else:
            next_state,end_of_trials = self.next_state(action)
            env_action = -1
        self.reward_ctr += reward
        self.env_reward_ctr += env_agent_reward
        if end_of_trials == 1:
            done = 1
            print('reward ctr {}, env reward ctr {}'.format(self.reward_ctr,self.env_reward_ctr))
            _ = self.env_reset()
        else:
            done = 0
        return reward,env_agent_reward,next_state,current_state,action,env_action,done

# Create the training function for the environment

STORAGE_MAX = 30000# originally 20000
def Gain_ExperienceDQN(env,agent,NumEpisodes,storage,timeDim,explore_ratio,device,trainFlag=True,return_actions = False,gamma=0.9,return_reps=False):
  agent.eval()
  if len(storage) > STORAGE_MAX:
    storage = storage[int(len(storage)-STORAGE_MAX):]
  all_qvects = []
  all_env_actions = []
  all_actions = []
  all_q_reps = []
  all_timeBins = []
  all_labels = []
  ##
  if return_reps == True:
    weights = []
    def patch_attention(m):
      forward_orig = m.forward

      def wrap(*args, **kwargs):
          kwargs['need_weights'] = True
          kwargs['average_attn_weights'] = False

          return forward_orig(*args, **kwargs)
      m.forward = wrap
    def get_weights():
      def hook(model, input, output):
          weights.append(output)
      return hook
    for module in agent.second_path.transformer_encoderT2.layers[1].modules():
      if isinstance(module, nn.MultiheadAttention):
          patch_attention(module)
          module.register_forward_hook(get_weights())
  for ijk in range(NumEpisodes):
    print('episode {}'.format(ijk+1))
    current_state= env.env_reset()
    current_done = 0
    # current_state = 0
    running_rewards = []
    running_env_rewards = []
    actions = []
    env_actions = []
    labels = []
    q_vects = []
    q_reps = []
    timeBins = []
    total_reward = 0
    # startPoints = []
    # agent.q_head[2].register_forward_hook(get_activation())
    tmp_state_length = 1
    while current_done != 1:
      # retrieve the action
      with torch.no_grad():
        qval_preds = agent(torch.from_numpy(current_state).float().to(device),torch.tensor([tmp_state_length]).to(device))#,torch.tensor([current_label]))
        q_vects.append(qval_preds.cpu().numpy())
      if explore_ratio != 1:
        action = np.random.choice([0,1],p=nn.functional.softmax(qval_preds[:,:],dim=1).reshape(-1).cpu().numpy())
      else:
        action = np.argmax(qval_preds[0,:].cpu().numpy())
      #
      # apply the action  reward,env_agent_reward,next_state,current_state,action,env_action,done
      reward,ea_reward,next_state,current_state0,action,env_action,current_done = env.take_action(action)
      next_state = np.expand_dims(next_state,axis=0)
      current_state0 = np.expand_dims(current_state0,axis=0)
      tmp_state_length = next_state.shape[2]
      with torch.no_grad():
        next_qval_preds = agent(torch.from_numpy(next_state).float().to(device),torch.tensor([tmp_state_length]).to(device))#,torch.tensor([next_label]))
        # next_qval_preds_env  = env.EnvAgent(torch.argmax(next_qval_preds,dim=1).reshape(1,-1))
        next_qval_preds_env  = env.EnvAgent(torch.from_numpy(next_state).float().to(device))
        #
      if action == 1 or action == 0 or tmp_state_length >= timeDim:#next_state.shape[2] < current_state0.shape[2]:
        actions.append(action)
        env_actions.append(env_action)
        # labels.append(current_label0)
        timeBins.append(current_state0.shape[2])
        activation_list = []
        trial_done = 1
      else:
        trial_done = 0
      target_value = reward + gamma*np.max(next_qval_preds[0,:].cpu().numpy())*(1-trial_done)
      target_env_value = ea_reward + gamma*np.max(next_qval_preds_env[0,:].cpu().numpy())*(1-trial_done)
      # compute the qvals
      running_rewards.append(reward)
      running_env_rewards.append(ea_reward)
      storage.append([reward,ea_reward,next_state,current_state,action,current_done,target_value,tmp_state_length,target_env_value,env_action])
      current_state  = next_state
      # current_label = next_label
    all_actions.append(actions)
    all_env_actions.append(env_actions)
    all_qvects.append(q_vects)
    all_q_reps.append(q_reps)
    all_timeBins.append(timeBins)
    # all_labels.append(labels)
  # print('total reward {}'.format(total_reward))
  if return_actions == True and return_reps == False:
    return storage,all_actions,all_qvects,all_q_reps,all_timeBins,running_rewards,running_env_rewards,all_env_actions
    #storage,all_actions,all_qvects,all_q_reps,all_timeBins,all_labels,running_rewards,running_env_rewards,all_env_actions
  if return_actions == True and return_reps == True:
    return storage,all_actions,all_qvects,all_q_reps,all_timeBins,weights,running_rewards,running_env_rewards,all_env_actions
    #storage,all_actions,all_qvects,all_q_reps,all_timeBins,weights,all_labels,running_rewards,running_env_rewards,all_env_actions
  else:
    return storage,running_rewards,running_env_rewards

def train_batch(trainData,agent,env_agent,optim,env_optim,device,gamma=0.99):
  batch_size = 200 #1 #200
  loss1_list,loss2_list,total_loss_list = [],[],[]
  total_loss_env_list = []
  agent.train()
  env_agent.train()
  buffer= ExperienceBuffer(trainData)
  loss_fn_value = nn.MSELoss()
  data_loader = torch.utils.data.DataLoader(buffer,batch_size=batch_size,collate_fn=buffer.collate_fn,shuffle=True,num_workers=0)# shuffle=False
  for ix,batch1 in enumerate(iter(data_loader)):
      batch_s,batch_rewards,batch_action,batch_qval,mask_s,batch_start,batch_env_rewards,batch_env_qval,batch_env_action = batch1
      val_out = agent(batch_s.squeeze(1).to(device),batch_start.to(device))
      # print('q val shape {} val shape {}'.format(batch_qval.shape,val_out[range(batch_action.shape[0]),batch_action.reshape(-1)].reshape(-1,1).shape))
      loss2 = loss_fn_value(val_out[range(batch_action.shape[0]),batch_action.reshape(-1)].reshape(-1,1).to(device),batch_qval.unsqueeze(1).to(device))
      total_loss=loss2
      total_loss_list.append(total_loss.item())
      total_loss.backward()
      optim.step()
      optim.zero_grad()
      # UPDATE THE ENV AGENT
      # val_out_env = env_agent(batch_action)
      val_out_env = env_agent(batch_s.squeeze(1).to(device))
      loss3 = loss_fn_value(val_out_env[range(val_out_env.shape[0]),batch_env_action.reshape(-1)].reshape(-1,1).to(device),batch_env_qval.unsqueeze(1).to(device))
      total_loss_env_list.append(loss3.item())
      loss3.backward()
      env_optim.step()
      env_optim.zero_grad()
  agent.eval()
  env_agent.eval()
  return np.mean(total_loss_list),np.mean(total_loss_env_list)

def test_batch(testData,agent,device,gamma=0.99):
      batch_size = 200 #1 #200
      loss1_list,loss2_list,total_loss_list = [],[],[]
      agent.eval()
      buffer= ExperienceBuffer(testData)
      loss_fn_value = nn.MSELoss()
      data_loader = torch.utils.data.DataLoader(buffer,batch_size=batch_size,collate_fn=buffer.collate_fn,shuffle=True,num_workers=0)# shuffle=False
      for ix,batch1 in enumerate(iter(data_loader)):
          batch_s,batch_rewards,batch_action,batch_qval,mask_s,batch_start,batch_env_rewards,batch_env_qval,batch_env_action = batch1
          val_out = agent(batch_s.squeeze(1).to(device),batch_start.to(device))#,batch_label)
          loss2 = loss_fn_value(val_out[range(val_out.shape[0]),batch_action].reshape(-1,1).to(device),batch_qval.unsqueeze(1).to(device))
          total_loss=loss2
          total_loss_list.append(total_loss.item())
      agent.train()
      return np.mean(total_loss_list)
