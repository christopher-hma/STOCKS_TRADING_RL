import numpy as np
import torch.nn as nn
import torch
import pandas as pd
import copy
import random
import torch.nn.functional as F

class ReplayBuffer():
    
    def __init__(self):
        
        super(ReplayBuffer, self).__init__()
        
        self.size = 0
        
        self.maxlen = 1000000
        
        self.cur_obs_queue = []
        
        self.action_queue = []
        
        self.next_obs_queue = []
        
        self.reward_queue = []
        
        self.done_queue = []
        
    def addExperience(self,obs,action,next_obs,reward,done):
        
        
        if self.size == self.maxlen:
            
           self.cur_obs_queue.pop(0)
        
           self.action_queue.pop(0)
        
           self.next_obs_queue.pop(0)
        
           self.reward_queue.pop(0)
        
           self.done_queue.pop(0) 
            
        else:
            
            self.size+=1
        
        self.cur_obs_queue.append(obs)
        
        self.action_queue.append(action)
        
        self.reward_queue.append(reward)
        
        self.next_obs_queue.append(next_obs)
        
        self.done_queue.append(done)
        
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

class Trajectory():
    
    def __init__(self):
        
        super(Trajectory, self).__init__()
        
        self.size = 0
        
        self.cur_obs_queue = []
        
        self.action_queue = []
        
        self.next_obs_queue = []
        
        self.reward_queue = []
        
        self.done_queue = []
        
        self.value_queue = []
        
        self.oldlogprob_queue = []
        
    def reset(self):
        
        self.size = 0
        
        self.cur_obs_queue.clear()
        
        self.action_queue.clear()
        
        self.next_obs_queue.clear()
        
        self.reward_queue.clear()
        
        self.done_queue.clear()
        
        self.value_queue.clear()
        
        self.oldlogprob_queue.clear()
        
    def push(self,obs,action,nobs,reward,done,value,logprob):
        
        self.size+=1
        
        self.cur_obs_queue.append(obs)
        
        self.action_queue.append(action)
        
        self.reward_queue.append(reward)
        
        self.next_obs_queue.append(nobs)
        
        self.done_queue.append(done)
        
        self.value_queue.append(value)
        
        self.oldlogprob_queue.append(logprob)  