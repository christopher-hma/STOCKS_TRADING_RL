import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import pandas as pd
from torch.distributions import MultivariateNormal
import copy

class Actor(nn.Module):
    
    def __init__(self,state_dim,mid_dim,num_actions):
        
        super(Actor,self).__init__()
        
        self.state_dim = state_dim        
        
        self.action_dim = num_actions
        
        self.mid_dim = mid_dim
        
        
        self.encoder = nn.Sequential(
           
             nn.Linear(self.state_dim,self.mid_dim),
             nn.ReLU(),            
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,self.action_dim)
            
        )   
    
    
    def forward(self,input):
        
        output = self.encoder(input).tanh()
        
        return output
        
        
    
class Critic(nn.Module):
    
    def __init__(self,state_dim,mid_dim,num_actions):
        
        super(Critic, self).__init__()
        
        self.state_dim = state_dim
        
        self.action_dim = num_actions
        
        self.mid_dim = mid_dim
        
        self.combined_dim = self.state_dim + self.action_dim
        
        self.encoder = nn.Sequential(
           
             nn.Linear(self.combined_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),            
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,1)
        
        )    
    
    def forward(self,state,action):
        
        
        combined_input = torch.cat((state,action), dim = -1)
        
        output = self.encoder(combined_input)
        
        return output    

class ActorTD3(nn.Module):
    
    def __init__(self,state_dim,mid_dim,num_actions):
        
        super(ActorTD3,self).__init__()
        
        self.state_dim = state_dim
        
        self.mid_dim = mid_dim
        
        self.num_actions = num_actions
        
        self.encoder = nn.Sequential(
        
            nn.Linear(state_dim,mid_dim),
            nn.ReLU(),
            
            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),
        
            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),
            
            nn.Linear(mid_dim,num_actions)
        
        )      
    
    def forward(self,input):
        
        output = self.encoder(input).tanh()
        
        return output

    
class CriticTD3(nn.Module):
    
    def __init__(self,state_dim,mid_dim,num_actions):
        
        super(CriticTD3, self).__init__()
        
        self.state_dim = state_dim
        
        self.mid_dim = mid_dim
        
        self.num_actions = num_actions
        
        self.encoder = nn.Sequential(
        
            nn.Linear(state_dim + num_actions,mid_dim),
            nn.ReLU(),
            
            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),
        
            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),

            nn.Linear(mid_dim,mid_dim),
            nn.ReLU(),
            
            nn.Linear(mid_dim,1)
        
        )      
    
    def forward(self,state,action):
        
        combined_input = torch.cat((state,action), dim = -1)
        
        output = self.encoder(combined_input)
        
        return output    
    
class ActorPPO(nn.Module):
    
    def __init__(self,state_dim,mid_dim,action_dim):
        
        super(ActorPPO, self).__init__()
        
        self.state_dim = state_dim
        
        self.action_dim = action_dim
        
        self.mid_dim = mid_dim
        
        action_std = 0.5
        
        self.encoder = nn.Sequential(
           
             nn.Linear(self.state_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,self.action_dim)
             
        )   
        
        
        self.action_var = nn.Parameter(torch.full((self.action_dim,), -action_std * action_std),requires_grad=True)
    
    def forward(self,input):
        
        mu = self.encoder(input)
    
        return mu,self.action_var.exp()


class CriticPPO(nn.Module):
    
    def __init__(self,state_dim,mid_dim):
        
        super(CriticPPO, self).__init__()
        
        self.state_dim = state_dim
        
        self.mid_dim = mid_dim
        
        self.encoder = nn.Sequential(
           
             nn.Linear(self.state_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),            
             nn.Linear(self.mid_dim,self.mid_dim),
             nn.ReLU(),
             nn.Linear(self.mid_dim,1)
        
        )    
    def forward(self,input):
        
        output = self.encoder(input)
        
        return output
    