import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import copy
import random
import torch.nn.functional as F


class Trajectory():
    
    def __init__(self):
        
        super(Trajectory, self).__init__()
        
        self.size = 0
        
        self.cur_obs_buffer = []
        
        self.action_buffer = []
        
        self.next_obs_buffer = []
        
        self.reward_buffer = []
        
        self.done_buffer = []
        
        self.shared_obs_buffer = []

        self.pred_val_buffer = []

        self.logprob_buffer = []
        
    def reset(self):
        
        self.size = 0
        
        self.cur_obs_buffer.clear()
        
        self.action_buffer.clear()
        
        self.next_obs_buffer.clear()
        
        self.reward_buffer.clear()
        
        self.done_buffer.clear()
        
        self.shared_obs_buffer.clear()
 
        self.pred_val_buffer.clear()

        self.logprob_buffer.clear()
        
    def insert_data(self,obs,shared_obs,action,reward,next_obs,done,value,logprob):
        
        self.size+=1
        
        self.cur_obs_buffer.append(obs)
        
        self.action_buffer.append(action)
        
        self.reward_buffer.append(reward)
        
        self.next_obs_buffer.append(next_obs)
        
        self.done_buffer.append(done)
        
        self.shared_obs_buffer.append(shared_obs)
        
        self.pred_val_buffer.append(value)

        self.logprob_buffer.append(logprob)
   

class ReplayBuffer():
    
    def __init__(self,state_dim,num_actions,num_agents):
        
        super(ReplayBuffer, self).__init__()
        
        self.size = 0
        
        self.maxlen = 10000
        
        self.cur_obs_queue = np.zeros((self.maxlen,num_agents,state_dim))
        
        self.action_queue = np.zeros((self.maxlen,num_agents,num_actions))
        
        self.next_obs_queue = np.zeros((self.maxlen,num_agents,state_dim))
        
        self.reward_queue = np.zeros((self.maxlen,num_agents))
        
        self.done_queue = np.zeros((self.maxlen,num_agents))
        
    def addExperience(self,obs,action,next_obs,reward,done):
        
        
        if self.size == self.maxlen:
            
           self.cur_obs_queue[:self.maxlen-1] = self.cur_obs_queue[1:]
        
           self.action_queue[:self.maxlen-1] = self.action_queue[1:]
        
           self.next_obs_queue[:self.maxlen-1] = self.next_obs_queue[1:]
        
           self.reward_queue[:self.maxlen-1] = self.reward_queue[1:]
        
           self.done_queue[:self.maxlen-1] = self.done_queue[1:]
            
        else:
            
            self.size+=1
        
        self.cur_obs_queue[self.size - 1] = obs
        
        self.action_queue[self.size - 1] = action
        
        self.reward_queue[self.size - 1] = reward
        
        self.next_obs_queue[self.size - 1] = next_obs
        
        self.done_queue[self.size - 1] = done
        
    def sample(self,batch_size):
        
        cur_obs_list = []
        
        action_list = []
        
        next_obs_list = []
        
        reward_list = []
        
        done_list = []
        
        mylist = [x for x in range(self.size)]

        indices = random.sample(mylist,batch_size)
        
        for index in indices:
            
            cur_obs_list.append(self.cur_obs_queue[index])
            
            action_list.append(self.action_queue[index])
            
            next_obs_list.append(self.next_obs_queue[index])
            
            reward_list.append(self.reward_queue[index])
            
            done_list.append(self.done_queue[index])
            
        return cur_obs_list,action_list,next_obs_list,reward_list,done_list



        
        