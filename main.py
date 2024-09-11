import numpy as np
import torch
from torch import nn
import h5py
from sklearn.preprocessing import MaxAbsScaler,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
import random
import copy
import os
from task_and_training_functions import DataEnv_NVPAug2_3Actions_All_Adv,Gain_ExperienceDQN,train_batch
from models import SRNN_Agent
from preproc_functions import Value_CS_ShiraData,StimID_Labeling,CorrectFlattening


if __name__ == '__main__':
    saveExt ='Outputs'
    if os.path.exists(saveExt) == False:
        os.mkdir(saveExt)
    jobid = int(os.environ.get('SLURM_ARRAY_TASK_ID'))-1
    sessionID = int(jobid%31) + 1
    if jobid <= 30:
      data = h5py.File('SmallWindows_Session_{}.h5'.format(sessionID))
      binnedSpikeCounts = np.transpose(data['trials'][:,:,:],(2,1,0))
      choice = data['choice'][:].T
      id1 = data['id1'][:].T
      id2 = data['id2'][:].T
      val_cs1 = Value_CS_ShiraData(id1)
      val_cs2 = Value_CS_ShiraData(id2)
      val1 = np.sum(val_cs1,axis=1)
      val2 = np.sum(val_cs2,axis=1)
      vdiff = val1-val2
    # define necessary parameters
    num_actions = 2
    learning_rate = 1e-4
    device = torch.device("cuda")
    batch_X,batch_y = [],[]
    testFlag = 1
    chkFlag = 2000
    printFlag = 2
    reward_list = []
    total_reward = 0
    embed_length = 20
    d_mod = 100
    stimids_pair = StimID_Labeling(id1[:,0] - id2[:,0]).reshape(-1,1).astype('int')
    #
    # initialize the environment
    shuffleFlag = True
    split_range = [0.2] #[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for qq in split_range:
      test_set_size = qq
      latentDim = d_mod
      tempTerm = 1
      trial_indx = np.array(range(binnedSpikeCounts.shape[0])).reshape(-1,1)
      allval = (np.sum(val_cs1,axis=1)-3)/2
      v_1 = np.sum(val_cs1,axis=1)
      v_2 = np.sum(val_cs2,axis=1)
      vdiff =  np.abs(v_1 - v_2)
      # idx_ = np.nonzero(vdiff != 0)[0]
      # idx__ = np.nonzero(vdiff == 0)[0]
      # trial_indx_ = idx_.reshape(-1,1)
      allval_ = allval#[idx_]
      vdiff_ = vdiff#[idx_]
      if shuffleFlag == True:
        train_X0,test_X0,train_y,test_y = train_test_split(trial_indx,allval_,test_size=test_set_size,stratify=vdiff_,shuffle=shuffleFlag,random_state=np.random.randint(100))
      else:
        train_X0,test_X0,train_y,test_y = train_test_split(trial_indx,allval_,test_size=test_set_size,shuffle=shuffleFlag)
      training_rewards_list,testing_rewards_list,training_rewards_list_NoNoise = [],[],[]
      training_rewards_list_env,testing_rewards_env_list = [],[]
      training_tstRewards_list_env,training_tstRewards_list = [],[]
      # test_y = np.concatenate((test_y,allval[idx__]),axis=0)
      trainingSplitInfo = [train_X0,test_X0,train_y,test_y]
      train_X0 = train_X0.reshape(-1,1)
      test_X0 = test_X0.reshape(-1,1)
      # test_X0 = np.concatenate((test_X0,trial_indx[idx__,:]),axis=0)
      trial_correctVect =[]
      test_vdiff = vdiff[test_X0[:,0]]
      #
      all_ct = train_X0[:,0]
      all_performance_details = []
      scaler = Pipeline([('scaler1',StandardScaler(with_mean=True,with_std=True)),('scaler2',MaxAbsScaler())]).fit(CorrectFlattening(binnedSpikeCounts[all_ct,:,:]).T)
      #
      N = binnedSpikeCounts.shape[1]
      timeDim = binnedSpikeCounts.shape[2]
      otherParams = [latentDim,tempTerm,timeDim,N]
      env = DataEnv_NVPAug2_3Actions_All_Adv(binnedSpikeCounts[all_ct,:,:],v_1[all_ct],v_2[all_ct],choice[all_ct,0],scaler,d_mod,otherParams,device,TotalSimExamples=200,ChoiceFlag=True)# False
      env.EnvAgent.to(device)
      # initialize the agent
      agent =SRNN_Agent(latentDim,tempTerm,timeDim,N,num_actions)
      agent = agent.to(device)
      storage = []
      # define the optimizer
      optim = torch.optim.Adam([p for p in agent.parameters() if p.requires_grad == True],lr=learning_rate) #,weight_decay=1e-8
      env_optim = torch.optim.Adam([p for p in env.EnvAgent.parameters() if p.requires_grad == True],lr=learning_rate)
      # start training ....
      warmup_period= 0
      explore_ratio = 0
      explore_ratio_update = 10000
      gamma_val = 0.001
      gen_train_ratio = 10
      gen_train_flag = True #
      trial_grp_size = gen_train_ratio
      NumIters = test_X0.shape[0]
      NumEpisodes = 5
      NumEpochs = 300
      for i in range(NumEpochs):
          # get data
          #
          storage,tmp_train_rewards,tmp_train_rewards_env = Gain_ExperienceDQN(env,agent,NumEpisodes,storage,timeDim,explore_ratio,device=device,gamma=gamma_val) #originally storage
          training_rewards_list.append(np.sum(tmp_train_rewards)/all_ct.shape[0])#tmp_train_rewards)
          training_rewards_list_env.append(np.sum(tmp_train_rewards_env)/all_ct.shape[0])
          # train the model
          tmp_storage = random.sample(storage,np.min([400,len(storage)]))
          loss_val1,loss_val1_env = train_batch(tmp_storage,agent,env.EnvAgent,optim,env_optim,device,gamma=gamma_val)
          #
          print('----train case (test style)---- ')
          test_env = DataEnv_NVPAug2_3Actions_All_Adv(binnedSpikeCounts[train_X0[:,0],:,:],v_1[train_X0[:,0]],v_2[train_X0[:,0]],choice[train_X0[:,0],0],scaler,\
          d_mod,otherParams,device,TotalSimExamples=train_X0.shape[0],ChoiceFlag=True,TrainFlag =False)
          test_env.EnvAgent = env.EnvAgent
          _,tmp_Traintest_rewards,tmp_Traintest_rewards_env = Gain_ExperienceDQN(test_env,agent,1,[],timeDim,device=device,explore_ratio = 1,trainFlag=False,gamma=gamma_val)
          training_tstRewards_list.append(np.sum(tmp_Traintest_rewards)/train_X0.shape[0])
          training_tstRewards_list_env.append(np.sum(tmp_Traintest_rewards_env)/train_X0.shape[0])
          print('------')
          print('----test case---- ')
          test_env = DataEnv_NVPAug2_3Actions_All_Adv(binnedSpikeCounts[test_X0[:,0],:,:],v_1[test_X0[:,0]],v_2[test_X0[:,0]],choice[test_X0[:,0],0],scaler,\
          d_mod,otherParams,device=device,TotalSimExamples=test_X0.shape[0],ChoiceFlag=True,TrainFlag =False)
          test_env.EnvAgent = env.EnvAgent
          #
          _,tmp_test_rewards,tmp_test_rewards_env = Gain_ExperienceDQN(test_env,agent,1,[],timeDim,device=device,explore_ratio = 1,trainFlag=False,gamma=gamma_val)
          testing_rewards_list.append(np.sum(tmp_test_rewards)/test_X0.shape[0])
          testing_rewards_env_list.append(np.sum(tmp_test_rewards_env)/test_X0.shape[0])
          print('------')
          # make checkpoints
          if i%chkFlag == 0 and i != 0:
              print('checkpoint here')
              savename_chkpt = '{}/{}/ValueBased_SharedControl_Session_{}_ITM_GenTrain_{}_CheckPoint.pt'.format(saveExt,saveFolder,sessionID,gen_train_flag)
              torch.save(agent.state_dict(), savename_chkpt)
      # env_assess_data = []
      # test_env = DataEnv_NVPAug2_3Actions_All(binnedSpikeCounts[test_X0[:,0],:,:],v_1[test_X0[:,0]],v_2[test_X0[:,0]],choice[test_X0[:,0],0],scaler,id1[test_X0[:,0],:],stimids_pair[test_X0[:,0],:])
      # tmp_agent = copy.deepcopy(agent)
      # #(env,agent,NumEpisodes,storage,timeDim,explore_ratio,trainFlag=True,return_actions = False,gamma=0.9,return_reps=False)
      # all_outs00 = Gain_ExperienceDQN(test_env,tmp_agent,1,[],timeDim,explore_ratio = 1,trainFlag=False,return_actions=True,return_reps=True,gamma=gamma_val)
      # env_assess_data.append(all_outs00)
      #
      savename_chkpt = '{}/AdvRL_BaseAgent_Session_{}_Split{}_ActingAgent.pt'.format(saveExt,sessionID,int(100*qq))
      torch.save(agent.cpu().state_dict(), savename_chkpt)
      savename_chkpt2 = '{}/AdvRL_BaseAgent_Session_{}_Split{}_EnvAgent.pt'.format(saveExt,sessionID,int(100*qq))
      torch.save(env.EnvAgent.cpu().state_dict(), savename_chkpt2)
      #
      pickle.dump([training_rewards_list,training_rewards_list_env,trainingSplitInfo,testing_rewards_list,testing_rewards_env_list,\
      training_tstRewards_list,training_tstRewards_list_env],open('{}/AdvRL_Agent_Session_{}_Iteration_{}_SimpleCase2_ModifiedArch.pickle'.format(saveExt,sessionID,int(qq*100)),'wb'))
