import math

import torch.nn.functional as F

import torch

import torch.nn as nn

import numpy as np

from torch.distributions import MultivariateNormal

from matplotlib import pyplot as plt

import torch.optim as optim

from empyrical import sharpe_ratio,sortino_ratio

import pandas as pd

from tqdm.auto import tqdm

from random_process import OrnsteinUhlenbeckProcess

from models import Critic, Actor, CriticTD3, ActorTD3, CriticPPO, ActorPPO

from environment import MultiStockTradeEnvironment

from multiagents.memory import ReplayBuffer, Trajectory

from stats import calculate_statistics,plot

class MADDPGAgent(nn.Module):

      def __init__(self,state_dim,mid_dim1,mid_dim2,num_actions,df,df_eval,df_test,tech,args):
        
        
          super(MADDPGAgent, self).__init__()
        

          self.state_dim = state_dim
        
          self.num_actions = num_actions
    
          self.num_agents = args.num_agents
        
          self.df = df
            
          self.df_eval = df_eval
        
          self.df_test = df_test
        
          self.critic = [Critic(state_dim * self.num_agents ,mid_dim1 * self.num_agents,num_actions * self.num_agents) for i in range(self.num_agents)]
        
          self.actor = [Actor(state_dim,mid_dim2,num_actions) for i in range(self.num_agents)]
        
          self.actor_target = [Actor(state_dim,mid_dim2,num_actions) for i in range(self.num_agents)]
        
          self.critic_target = [Critic(state_dim * self.num_agents , mid_dim1 * self.num_agents,num_actions * self.num_agents) for i in range(self.num_agents)]
         
          self.clone_list_models(from_model_list = self.critic, to_model_list = self.critic_target)
        
          self.clone_list_models(from_model_list = self.actor, to_model_list = self.actor_target)
        
          self.lr = 2 ** -14
        
          self.critic_optimizer = [torch.optim.Adam(self.critic[i].parameters(), lr=self.lr, eps=1e-4) for i in range(self.num_agents)]
        
          self.actor_optimizer = [torch.optim.Adam(self.actor[i].parameters(),lr=self.lr, eps=1e-4) for i in range(self.num_agents)]
        
          self.NUM_STOCKS = num_actions       
        
          self.env = MultiStockTradeEnvironment(num_actions,self.num_agents,df,df_eval,df_test,tech,args.is_turbulence)
        
          self.replay_buffer = ReplayBuffer(state_dim,num_actions,self.num_agents)
               
          self.gamma = 0.99
        
          self.batchsize = args.batchsize
        
          self.random_seed = 0
      
          np.random.seed(self.random_seed)
        
          torch.manual_seed(self.random_seed)
        
          self.policy_update = 2

          self.random_process = OrnsteinUhlenbeckProcess(
            size=num_actions, theta=0.15, mu=0, sigma=0.2
          )
        
      def clone_list_models(self,from_model_list, to_model_list):
                
          for from_model, to_model in zip(from_model_list,to_model_list):
            
              self.clone_model(from_model,to_model)      
    
      
      def clone_model(self,from_model, to_model):
        
          for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            
              to_model.data.copy_(from_model.data.clone())    

      def net_update(self,from_model,to_model):
        
          tau = 2 ** -8
        
          for target_param, local_param in zip(to_model.parameters(), from_model.parameters()):
            
              target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
                       
      def optimizer_update(self, optimizer, network, loss):
        
          optimizer.zero_grad() #reset gradients to 0
        
          loss.backward(retain_graph=False) #this calculates the gradients
        
          optimizer.step()

      def step(self,obs,agent_idx):
        
          state = obs.copy()
            
          state = torch.from_numpy(state).float()
        
          action = self.actor[agent_idx](state)
        
          action += torch.from_numpy(self.random_process.sample()).float()
        
          return action
  
      def evaluate(self,args):

          statistics = []
        
          reward_ = list()
          
          if args.is_test:

             self.load_weights(args.output)
            
          for episode in range(args.validate_episodes):
        
              t = 0
         
              if args.is_test:
         
                T = len(self.df_test.index.unique().tolist())

              else:
                
                T = len(self.df_eval.index.unique().tolist())
        
              rewards = 0 
            
              total_asset_value = list()
          
              total_asset_value.append(self.env.budget * 3 + 1)
        
              if args.is_test:
 
                 obs = self.env.test_reset_()

              else:
        
                 obs = self.env.eval_reset_()
        
              for t in range(T-1):
                 
                  obs1 = obs.copy() 
                
                  action_list = []

                  for i in range(self.num_agents):

                      obs1[i] = obs1[i] * (2 ** -6)

                      obs1[i][0] = obs1[i][0] * (2 ** -6)
                  
            
                  for m in range(self.num_agents):
                    
                      actions = self.step(obs1[m],m)
        
                      actions = actions.squeeze()
            
                      actions = actions.detach().numpy()   
                
                      action_list.append(actions)

                  if args.is_test:
      
                     obs, reward, done, next_time = self.env.step(action_list,t,T,is_test = True)

                  else:

                     obs, reward, done, next_time = self.env.step(action_list,t,T,is_eval = True)

        
                  rewards += sum(reward)               
                
                  total_asset_value.append(total_asset_value[len(total_asset_value)-1] + sum(reward))
                    
              print(rewards)
            
              sharpe_ratio,sortino_ratio,mdd = calculate_statistics(total_asset_value)

              statistics.append((sharpe_ratio,sortino_ratio,mdd))
  
              reward_.append(rewards) 
                        

          res = [sum(x)/ len(statistics) for x in zip(*statistics)] 
     
          out = (res[0],res[1],res[2])    
            
          return out,np.mean(reward_),total_asset_value  

      def run(self,episode_length,t):
        
          finish = False
        
          T = len(self.df.index.unique().tolist())
        
          total_steps = 0 
        
          rewards = 0             
            
          obs = [self.env.get_state(agent_idx,is_eval = False,is_test = False).copy() for agent_idx in range(self.num_agents)]

          next_obs = obs
            
          while not finish:

              total_steps += 1  

              trajectory = list()
                
              action_list = []
                
              obs1 = obs.copy()

              for i in range(self.num_agents):

                  obs1[i] = obs1[i] * (2 ** -6)

                  obs1[i][0] = obs1[i][0] * (2 ** -6)

              for agent_idx in range(self.num_agents):
                    
                 #retrieve next action 

                 actions = self.step(obs1[agent_idx],agent_idx)
            
                 actions = actions.squeeze()
            
                 actions = actions.detach().numpy()
                
                 action_list.append(actions)
                
              next_obs, reward, done, next_time = self.env.step(action_list,t,T)

              rewards += sum(reward)

              for rew in reward:

                  rew =  rew * (2 ** -5)
                
              nobs1 = next_obs.copy()

              for i in range(self.num_agents):

                  nobs1[i] = nobs1[i] * (2 ** -6)

                  nobs1[i][0] = nobs1[i][0] * (2 ** -6)

              trajectory.append((obs1,action_list,nobs1,reward,done))    

              cstate = np.asarray([trajectory[0][0]]).squeeze()

              nstate = np.asarray([trajectory[0][2]]).squeeze()

              act = np.asarray([trajectory[0][1]]).squeeze()

              reward_ = np.asarray([trajectory[0][3]]).squeeze()

              done_ = np.asarray([trajectory[0][4]]).squeeze()

              self.replay_buffer.addExperience(cstate,act,nstate,reward_,done_)

              obs = next_obs
                
              if total_steps == episode_length:
                
                     finish = True
              
              t = next_time      
        
          self.train_net()
                      
          return t   



      def train_net(self):

          num_iterations = (math.floor(self.replay_buffer.size/self.batchsize))* 8    

          for _ in tqdm(range(num_iterations)):
                
             indices = torch.randint(self.replay_buffer.size,size=(self.batchsize,), requires_grad=False).tolist()
              
             for agent_idx in range(self.num_agents):     

                 self.train_step(_,agent_idx,indices)    

             for agent_idx in range(self.num_agents):
             
                 self.net_update(from_model = self.critic[agent_idx], to_model = self.critic_target[agent_idx])
                                      
                 self.net_update(from_model = self.actor[agent_idx], to_model = self.actor_target[agent_idx])    


      def train_step(self,_,agent_idx,indices):
        
          action_std = 0.2
       
          with torch.no_grad():
                  
                 
                 next_obs = [self.replay_buffer.next_obs_queue[index] for index in indices]

                 next_obs = np.asarray(next_obs)

                 next_obs_ten = torch.from_numpy(next_obs).float() 

                 next_predict_action = [self.actor_target[m](next_obs_ten[:,m]) for m in range(self.num_agents)]
                     
                 next_obs_ten = next_obs_ten.reshape(next_obs_ten.shape[0],next_obs_ten.shape[1] * next_obs_ten.shape[2])                        
                 
                    
                 cur_obs = [self.replay_buffer.cur_obs_queue[index] for index in indices]

                 cur_obs = np.asarray(cur_obs)
                    
                 cur_obs_ten = torch.from_numpy(cur_obs).float() 
                
                 cur_obs_ten = cur_obs_ten.reshape(cur_obs_ten.shape[0],cur_obs_ten.shape[1] * cur_obs_ten.shape[2])                        
                                 

                 cur_obs_ten_c = cur_obs_ten.clone()                    
                                         
                 next_predict_action = torch.stack(next_predict_action)

                 next_predict_action = next_predict_action.transpose(1,0) 
                 
                 next_predict_action = next_predict_action.reshape(next_predict_action.shape[0],next_predict_action.shape[1] * next_predict_action.shape[2])                        
                 
                   
                 
                 Qval_predict = self.critic_target[agent_idx](next_obs_ten,next_predict_action)
                                
                 reward_list = [self.replay_buffer.reward_queue[index] for index in indices]

                 reward_ = np.asarray(reward_list)
 
                 rew_ten = torch.from_numpy(reward_).float()
                    
                 

                 done_list = [self.replay_buffer.done_queue[index] for index in indices]

                 done_ = np.asarray(done_list)

                 done_ten = torch.from_numpy(done_).float()   
                    
                
                
                 # calculate predicted Q value 
  
                 predict_value = rew_ten[:,agent_idx].unsqueeze(-1) + self.gamma * Qval_predict * (1-done_ten[:,agent_idx]).unsqueeze(-1)
                
                 action_list = [self.replay_buffer.action_queue[index] for index in indices]

                 action_array = np.asarray(action_list)

                 action_ten = torch.from_numpy(action_array).float()

                 action_ten = action_ten.reshape(action_ten.shape[0],action_ten.shape[1] * action_ten.shape[2])                        
                         
                 action_ten_c = action_ten.clone()
                
                  
          # calculate actual Q value
                 
          actual_value = self.critic[agent_idx](cur_obs_ten,action_ten)              
                 
          # critic loss
                 
          critic_loss = F.mse_loss(actual_value,predict_value).mean()
                       
          self.optimizer_update(self.critic_optimizer[agent_idx],self.critic[agent_idx], critic_loss)
            

        
          # actor loss
          cur_pred_action = self.actor[agent_idx](cur_obs_ten_c[:,agent_idx * self.state_dim:(agent_idx + 1) * self.state_dim])
                                 
          action_ten_c[:,agent_idx * self.num_actions : (agent_idx + 1) * self.num_actions] = cur_pred_action
                      
          actor_loss = -1 * self.critic[agent_idx](cur_obs_ten_c,action_ten_c)
        
          actor_loss = actor_loss.mean()
                                   
          self.optimizer_update(self.actor_optimizer[agent_idx],self.actor[agent_idx], actor_loss) 
                                            
      def eval(self):
    
          for agent_idx in range(self.num_agents):
        
              self.actor[agent_idx].eval()
        
              self.actor_target[agent_idx].eval()
        
              self.critic[agent_idx].eval()
        
              self.critic_target[agent_idx].eval()

      def train(self):
       
          for agent_idx in range(self.num_agents):
        
              self.actor[agent_idx].train()
        
              self.actor_target[agent_idx].train()
        
              self.critic[agent_idx].train()
        
              self.critic_target[agent_idx].train()     


      def load_weights(self, output):
    
          for agent_idx in range(self.num_agents):
      
              self.actor[agent_idx].load_state_dict(torch.load("{}/actor{}.pkl".format(output,agent_idx)))
    
              self.actor_target[agent_idx].load_state_dict(
            
                   torch.load("{}/actor_target{}.pkl".format(output,agent_idx))
        
              )
        
              self.critic[agent_idx].load_state_dict(torch.load("{}/critic{}.pkl".format(output,agent_idx)))
        
              self.critic_target[agent_idx].load_state_dict(
            
                   torch.load("{}/critic_target{}.pkl".format(output,agent_idx))
        
              )

             
      def save_model(self, output):
        
          for agent_idx in range(self.num_agents):
      
              torch.save(self.actor[agent_idx].state_dict(), "{}/actor{}.pkl".format(output,agent_idx))
        
              torch.save(self.critic[agent_idx].state_dict(), "{}/critic{}.pkl".format(output,agent_idx))
        
              torch.save(self.actor_target[agent_idx].state_dict(), "{}/actor_target{}.pkl".format(output,agent_idx))
        
              torch.save(self.critic_target[agent_idx].state_dict(), "{}/critic_target{}.pkl".format(output,agent_idx))

            


class MATD3Agent(nn.Module):

      def __init__(self,state_dim,mid_dim1,mid_dim2,num_actions,df,df_eval,df_test,tech,args):
        
        
          super(MATD3Agent, self).__init__()       

          self.state_dim = state_dim
        
          self.num_actions = num_actions
        
          self.num_agents = args.num_agents
            
          self.df = df
        
          self.df_eval = df_eval
            
          self.df_test = df_test
        
          self.critic1 = [CriticTD3(state_dim * self.num_agents ,mid_dim1 * self.num_agents,num_actions * self.num_agents) for i in range(self.num_agents)]
        
          self.critic2 =[CriticTD3(state_dim * self.num_agents,mid_dim1 * self.num_agents,num_actions * self.num_agents) for i in range(self.num_agents)]
        
          self.actor = [ActorTD3(state_dim,mid_dim2,num_actions) for i in range(self.num_agents)]
        
          self.actor_target = [ActorTD3(state_dim,mid_dim2,num_actions) for i in range(self.num_agents)]
        
          self.critic1_target = [CriticTD3(state_dim * self.num_agents , mid_dim1 * self.num_agents,num_actions * self.num_agents) for i in range(self.num_agents)]
        
          self.critic2_target = [CriticTD3(state_dim * self.num_agents, mid_dim1 * self.num_agents,num_actions * self.num_agents) for i in range(self.num_agents)]
            
          self.clone_list_models(from_model_list = self.critic1, to_model_list = self.critic1_target)
        
          self.clone_list_models(from_model_list = self.critic2, to_model_list = self.critic2_target)
        
          self.clone_list_models(from_model_list = self.actor, to_model_list = self.actor_target)
        
          self.lr = 2 ** -14
        
          self.critic1_optimizer = [torch.optim.Adam(self.critic1[i].parameters(), lr=self.lr, eps=1e-4) for i in range(self.num_agents)]
        
          self.critic2_optimizer = [torch.optim.Adam(self.critic2[i].parameters(), lr=self.lr, eps=1e-4) for i in range(self.num_agents)]
        
          self.actor_optimizer = [torch.optim.Adam(self.actor[i].parameters(),lr=self.lr, eps=1e-4) for i in range(self.num_agents)]
        
          self.NUM_STOCKS = num_actions       
        
          self.env = MultiStockTradeEnvironment(num_actions,self.num_agents,df,df_eval,df_test,tech,args.is_turbulence)
        
          self.replay_buffer = ReplayBuffer(state_dim,num_actions,self.num_agents)
               
          self.gamma = 0.99
        
          self.batchsize = args.batchsize
        
          self.random_seed = 0
      
          np.random.seed(self.random_seed)
        
          torch.manual_seed(self.random_seed)
        
          self.policy_update = 2


      def clone_list_models(self,from_model_list, to_model_list):
                
          for from_model, to_model in zip(from_model_list,to_model_list):
            
              self.clone_model(from_model,to_model)      
    
      
      def clone_model(self,from_model, to_model):
        
          for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            
              to_model.data.copy_(from_model.data.clone())    


      def net_update(self,from_model,to_model):
        
          tau = 2 ** -8
        
          for target_param, local_param in zip(to_model.parameters(), from_model.parameters()):
            
              target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
            
      def optimizer_update(self, optimizer, network, loss):
        
          optimizer.zero_grad() #reset gradients to 0
        
          loss.backward(retain_graph=False) #this calculates the gradients
        
          optimizer.step()

      
      def step(self,obs,agent_idx):
        
          state = obs.copy()
            
          state = torch.from_numpy(state).float()
        
          action = self.actor[agent_idx](state)
        
          action = (action + torch.randn_like(action) * 0.2).clamp(-1, 1)
        
          return action
      
      def evaluate(self,args):

          statistics = []
        
          reward_ = list()
          
          if args.is_test:

             self.load_weights(args.output)
            
          for episode in range(args.validate_episodes):
        
              t = 0
         
              if args.is_test:
         
                T = len(self.df_test.index.unique().tolist())

              else:
                
                T = len(self.df_eval.index.unique().tolist())
        
              rewards = 0 
            
              total_asset_value = list()
          
              total_asset_value.append(self.env.budget * 3 + 1)

              if args.is_test:
 
                 obs = self.env.test_reset_()

              else:
        
                 obs = self.env.eval_reset_()
        
              for t in range(T-1):
                 
                  obs1 = obs.copy() 
                
                  action_list = []

                  for i in range(self.num_agents):

                      obs1[i] = obs1[i] * (2 ** -6)

                      obs1[i][0] = obs1[i][0] * (2 ** -6)
                  
            
                  for m in range(self.num_agents):
                    
                      actions = self.step(obs1[m],m)
        
                      actions = actions.squeeze()
            
                      actions = actions.detach().numpy()   
                
                      action_list.append(actions)
      
                  if args.is_test:
      
                     obs, reward, done, next_time = self.env.step(action_list,t,T,is_test = True)

                  else:

                     obs, reward, done, next_time = self.env.step(action_list,t,T,is_eval = True)

        
                  rewards += sum(reward)               
                
                  total_asset_value.append(total_asset_value[len(total_asset_value)-1] + sum(reward))
                    
              print(rewards)
            
              sharpe_ratio,sortino_ratio,mdd = calculate_statistics(total_asset_value)

              statistics.append((sharpe_ratio,sortino_ratio,mdd))
      
              reward_.append(rewards) 
            
          res = [sum(x)/ len(statistics) for x in zip(*statistics)] 
     
          out = (res[0],res[1],res[2])    
  
          return out,np.mean(reward_),total_asset_value  

      def run(self,episode_length,t):
        
          finish = False
        
          T = len(self.df.index.unique().tolist())
        
          total_steps = 0 
        
          rewards = 0             
           
          obs = [self.env.get_state(agent_idx,is_eval = False,is_test = False).copy() for agent_idx in range(self.num_agents)]

          next_obs = obs
            
          while not finish:
                              
              total_steps += 1  

              trajectory = list()
                
              action_list = []
             
              obs1 = obs.copy()

              for i in range(self.num_agents):

                  obs1[i] = obs1[i] * (2 ** -6)

                  obs1[i][0] = obs1[i][0] * (2 ** -6)


              for agent_idx in range(self.num_agents):
                    
                 #retrieve next action

                 actions = self.step(obs1[agent_idx],agent_idx)
            
                 actions = actions.squeeze()
            
                 actions = actions.detach().numpy()
                
                 action_list.append(actions)
                
              next_obs, reward, done, next_time = self.env.step(action_list,t,T)

              nobs1 = next_obs.copy()

              for i in range(self.num_agents):

                  nobs1[i] = nobs1[i] * (2 ** -6)

                  nobs1[i][0] = nobs1[i][0] * (2 ** -6)

             
              rewards += sum(reward)

              for rew in reward:

                  rew =  rew * (2 ** -5)
      
              trajectory.append((obs1,action_list,nobs1,reward,done))  

              cstate = np.asarray([trajectory[0][0]]).squeeze()

              nstate = np.asarray([trajectory[0][2]]).squeeze()

              act = np.asarray([trajectory[0][1]]).squeeze()

              reward_ = np.asarray([trajectory[0][3]]).squeeze()

              done_ = np.asarray([trajectory[0][4]]).squeeze()

              self.replay_buffer.addExperience(cstate,act,nstate,reward_,done_)

              obs = next_obs
                
              if total_steps == episode_length:
                
                     finish = True
              
              t = next_time      
        
          self.train_net()
                      
          return t
   
    
      def train_net(self):

          num_iterations = (math.floor(self.replay_buffer.size/self.batchsize)) * 8    

          for _ in tqdm(range(num_iterations)):
                
             indices = torch.randint(self.replay_buffer.size,size=(self.batchsize,), requires_grad=False).tolist()
              
             for agent_idx in range(self.num_agents):     

                 self.train_step(_,agent_idx,indices)    
                                    
             if _ % self.policy_update == 0:
                
                 for agent_idx in range(self.num_agents):
                                            
                      self.net_update(from_model = self.critic1[agent_idx], to_model = self.critic1_target[agent_idx])
                             
                      self.net_update(from_model = self.critic2[agent_idx], to_model = self.critic2_target[agent_idx])
                   
                      self.net_update(from_model = self.actor[agent_idx], to_model = self.actor_target[agent_idx])         


      def train_step(self,_,agent_idx,indices):
        
          action_std = 0.2
       
          with torch.no_grad():
                                 
                 next_obs = [self.replay_buffer.next_obs_queue[index] for index in indices]

                 next_obs = np.asarray(next_obs)

                 next_obs_ten = torch.from_numpy(next_obs).float() 

                 next_predict_action = [self.actor_target[m](next_obs_ten[:,m]) for m in range(self.num_agents)]
                     
                 next_obs_ten = next_obs_ten.reshape(next_obs_ten.shape[0],next_obs_ten.shape[1] * next_obs_ten.shape[2])                        
                 

                 
                 
                 cur_list = [self.replay_buffer.cur_obs_queue[index] for index in indices]

                 cur_obs = np.asarray(cur_list)
                    
                 cur_obs_ten = torch.from_numpy(cur_obs).float() 
                
                 cur_obs_ten = cur_obs_ten.reshape(cur_obs_ten.shape[0],cur_obs_ten.shape[1] * cur_obs_ten.shape[2])                        
                 
                 

                 cur_obs_ten_c = cur_obs_ten.clone()
                    
                                         
                 next_predict_action = torch.stack(next_predict_action)

                 next_predict_action = next_predict_action.transpose(1,0) 
                 
                 next_predict_action = next_predict_action.reshape(next_predict_action.shape[0],next_predict_action.shape[1] * next_predict_action.shape[2])                        
                 
                 
                 noise = (torch.randn_like(next_predict_action) * action_std).clamp(-0.5, 0.5)
        
                 next_predict_action = (next_predict_action + noise).clamp(-1.0, 1.0)

                   
                 
                 Qval_predict1 = self.critic1_target[agent_idx](next_obs_ten,next_predict_action)
                
                 Qval_predict2 = self.critic2_target[agent_idx](next_obs_ten,next_predict_action)
                    
                 Qval_predict = torch.min(Qval_predict1,Qval_predict2) 
                
                
                
                 reward_list = [self.replay_buffer.reward_queue[index] for index in indices]

                 rarray = np.asarray(reward_list)
 
                 rew_ten = torch.from_numpy(rarray).float()
                    
                 

                 done_list = [self.replay_buffer.done_queue[index] for index in indices]

                 darray = np.asarray(done_list)

                 done_ten = torch.from_numpy(darray).float()   
                    
                
                
                 # calculate predicted Q value 
  
                 predict_value = rew_ten[:,agent_idx].unsqueeze(-1) + self.gamma * Qval_predict * (1-done_ten[:,agent_idx]).unsqueeze(-1)
                
                 action_list = [self.replay_buffer.action_queue[index] for index in indices]

                 action_array = np.asarray(action_list)

                 action_ten = torch.from_numpy(action_array).float()

                 action_ten = action_ten.reshape(action_ten.shape[0],action_ten.shape[1] * action_ten.shape[2])                        
                         
                 action_ten_c = action_ten.clone()
                
                  
          # calculate actual Q value
                 
          actual_value1 = self.critic1[agent_idx](cur_obs_ten,action_ten)  
                
          actual_value2 = self.critic2[agent_idx](cur_obs_ten,action_ten)  
                    
        
                 
          # critic loss
                 
          critic_loss1 = F.mse_loss(actual_value1,predict_value).mean()
        
          critic_loss2 = F.mse_loss(actual_value2,predict_value).mean()
        
               
        
          self.optimizer_update(self.critic1_optimizer[agent_idx],self.critic1[agent_idx], critic_loss1)
            
          self.optimizer_update(self.critic2_optimizer[agent_idx],self.critic2[agent_idx], critic_loss2)
        
        
          # actor loss
          cur_pred_action = self.actor[agent_idx](cur_obs_ten_c[:,agent_idx * self.state_dim:(agent_idx + 1) * self.state_dim])
                                 
          action_ten_c[:,agent_idx * self.num_actions : (agent_idx + 1) * self.num_actions] = cur_pred_action
                      
          actor_loss = -1 * self.critic1[agent_idx](cur_obs_ten_c,action_ten_c)
        
          actor_loss = actor_loss.mean()
                                    
          self.optimizer_update(self.actor_optimizer[agent_idx],self.actor[agent_idx], actor_loss) 
        
        
      def eval(self):
     
          for agent_idx in range(self.num_agents):
        
              self.actor[agent_idx].eval()
        
              self.actor_target[agent_idx].eval()
        
              self.critic1[agent_idx].eval()
        
              self.critic1_target[agent_idx].eval()
        
              self.critic2[agent_idx].eval()
        
              self.critic2_target[agent_idx].eval()

      def train(self):
    
          for agent_idx in range(self.num_agents):
        
              self.actor[agent_idx].train()
        
              self.actor_target[agent_idx].train()
        
              self.critic1[agent_idx].train()
        
              self.critic1_target[agent_idx].train()
        
              self.critic2[agent_idx].train()
        
              self.critic2_target[agent_idx].train()
        
                 
      def load_weights(self, output):
    
          for agent_idx in range(self.num_agents):
      
              self.actor[agent_idx].load_state_dict(torch.load("{}/actor{}.pkl".format(output,agent_idx)))
    
              self.actor_target[agent_idx].load_state_dict(
            
                   torch.load("{}/actor_target{}.pkl".format(output,agent_idx))
        
              )
        
              self.critic1[agent_idx].load_state_dict(torch.load("{}/critic1_{}.pkl".format(output,agent_idx)))
        
              self.critic1_target[agent_idx].load_state_dict(
            
                   torch.load("{}/critic1_target{}.pkl".format(output,agent_idx))
        
              )

              self.critic2[agent_idx].load_state_dict(torch.load("{}/critic2_{}.pkl".format(output,agent_idx)))
        
              self.critic2_target[agent_idx].load_state_dict(
            
                   torch.load("{}/critic2_target{}.pkl".format(output,agent_idx))
        
              )

      def save_model(self, output):
        
          for agent_idx in range(self.num_agents):
      
              torch.save(self.actor[agent_idx].state_dict(), "{}/actor{}.pkl".format(output,agent_idx))
        
              torch.save(self.critic1[agent_idx].state_dict(), "{}/critic1_{}.pkl".format(output,agent_idx))

              torch.save(self.critic2[agent_idx].state_dict(), "{}/critic2_{}.pkl".format(output,agent_idx))
        
              torch.save(self.actor_target[agent_idx].state_dict(), "{}/actor_target{}.pkl".format(output,agent_idx))
        
              torch.save(self.critic1_target[agent_idx].state_dict(), "{}/critic1_target{}.pkl".format(output,agent_idx))

              torch.save(self.critic2_target[agent_idx].state_dict(), "{}/critic2_target{}.pkl".format(output,agent_idx))

               

class MAPPOAgent(nn.Module):
    
    def __init__(self,state_dim,mid_dim1,mid_dim2,num_actions,df,df_eval,df_test,tech,args):
        
        super(MAPPOAgent, self).__init__()
        
        self.state_dim = state_dim
        
        self.num_actions = num_actions
        
        self.num_agents = args.num_agents
        
        self.df = df
        
        self.df_eval = df_eval
        
        self.df_test = df_test
        
        self.critic = [CriticPPO(state_dim,mid_dim1 * self.num_agents,num_actions * (self.num_agents-1)) for i in range(self.num_agents)]
        
        self.actor = [ActorPPO(state_dim,mid_dim2,num_actions) for i in range(self.num_agents)]
        
        self.actor_old = [ActorPPO(state_dim,mid_dim2,num_actions) for i in range(self.num_agents)]
        
        self.critic_old = [CriticPPO(state_dim,mid_dim1 * self.num_agents,num_actions * (self.num_agents-1)) for i in range(self.num_agents)]
        
        self.clone_list_models(from_model_list = self.critic, to_model_list = self.critic_old)
        
        self.clone_list_models(from_model_list = self.actor, to_model_list = self.actor_old)
        
        self.lr = 2 ** -14
        
        self.critic_optimizer = [torch.optim.Adam(self.critic[i].parameters(),lr=self.lr, eps=1e-4) for i in range(self.num_agents)]
        
        self.actor_optimizer = [torch.optim.Adam(self.actor[i].parameters(),lr=self.lr, eps=1e-4) for i in range(self.num_agents)]
        
        self.NUM_STOCKS = num_actions
        
        self.gamma = 0.99
        
        self.batchsize = args.batchsize
        
        self.random_seed = 0        
        
        self.replay_buffer = [Trajectory() for i in range(self.num_agents)]
        
        self.env = MultiStockTradeEnvironment(num_actions,self.num_agents,df,df_eval,df_test,tech,args.is_turbulence)
        
        self.epsilon = 0.2
                
        np.random.seed(self.random_seed)
        
        torch.manual_seed(self.random_seed)
              
        self.last_value = [0 for i in range(self.num_agents)]

        self.rewards = []
        
        self.advantage = []

        self.clip_param = 0.2
    
    def clone_list_models(self,from_model_list, to_model_list):
                
        for from_model, to_model in zip(from_model_list,to_model_list):
            
            self.clone_model(from_model,to_model)      
            
    
    def clone_model(self,from_model, to_model):
        
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            
            to_model.data.copy_(from_model.data.clone())             
    
    
    def net_update(self,from_model,to_model,tau):
      
        for target_param, local_param in zip(to_model.parameters(), from_model.parameters()):
            
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            
            
    def optimizer_update(self, optimizer, network, loss):
        
        optimizer.zero_grad() #reset gradients to 0
        
        loss.backward(retain_graph=False) #this calculates the gradients
        
        optimizer.step()   
    

    def evaluate(self,args):

          statistics = []
        
          reward_ = list()
          
          sharpe_ = list()
            
          if args.is_test:

             self.load_weights(args.output)

          for episode in range(args.validate_episodes):
        
              t = 0
         
              if args.is_test:
         
                T = len(self.df_test.index.unique().tolist())

              else:
                
                T = len(self.df_eval.index.unique().tolist())
        
              rewards = 0 
            
              total_asset_value = list()
          
              total_asset_value.append(self.env.budget * 3 + 1)
        
              if args.is_test:
 
                 obs = self.env.test_reset_()

              else:
        
                 obs = self.env.eval_reset_()
        
              for t in range(T-1):
                 
                  obs1 = obs.copy() 
                
                  action_list = []

                  for i in range(self.num_agents):

                      obs1[i] = obs1[i] * (2 ** -6)

                      obs1[i][0] = obs1[i][0] * (2 ** -6)
                  
            
                  for m in range(self.num_agents):
                    
                      actions = self.step(obs1[m],m)
                        
                      actions = actions.tanh()
        
                      actions = actions.squeeze()
            
                      actions = actions.detach().numpy()   
                
                      action_list.append(actions)
      
                  if args.is_test:
      
                     obs, reward, done, next_time = self.env.step(action_list,t,T,is_test = True)

                  else:

                     obs, reward, done, next_time = self.env.step(action_list,t,T,is_eval = True)

        
                  rewards += sum(reward)               
                
                  total_asset_value.append(total_asset_value[len(total_asset_value)-1] + sum(reward))
                    
              print(rewards)
            
              sharpe_ratio,sortino_ratio,mdd = calculate_statistics(total_asset_value)

              statistics.append((sharpe_ratio,sortino_ratio,mdd))

              reward_.append(rewards) 
            
              
          res = [sum(x)/ len(statistics) for x in zip(*statistics)] 
     
          out = (res[0],res[1],res[2]) 
    
          return out,np.mean(reward_),total_asset_value

    def clear(self):

        for buf in self.replay_buffer:

            buf.reset()
            
    def run(self,episode_length,t):
        
          finish = False
        
          T = len(self.df.index.unique().tolist())
        
          total_steps = 0 
        
          rewards = 0             
            
          obs = [self.env.get_state(agent_idx,is_eval = False,is_test = False).copy() for agent_idx in range(self.num_agents)]
        
          self.clear()
            
          while not finish:
                                         
              total_steps += 1  

              trajectory = list()
                
              action_list = []
            
              action_list1 = []
                
              obs1 = obs.copy()

              for i in range(self.num_agents):

                  obs1[i] = obs1[i] * (2 ** -6)

                  obs1[i][0] = obs1[i][0] * (2 ** -6)

            
              for agent_idx in range(self.num_agents):
                    
                 #retrieve next action

                 actions = self.step(obs1[agent_idx],agent_idx)
                    
                 actions1 = actions.clone()
                    
                 actions = actions.tanh().squeeze().detach().numpy()
                
                 actions1 = actions1.squeeze().detach().numpy()
            
                 action_list.append(actions)
                    
                 action_list1.append(actions1)
                
              next_obs, reward, done, next_time = self.env.step(action_list,t,T)

              rewards += sum(reward)

              for rew in reward:

                  rew =  rew * (2 ** -5)

                
              self.insert_data(obs1,action_list1,next_obs,reward,done)
                
              obs = next_obs
                
              if total_steps == episode_length:
                
                     finish = True
              
              t = next_time      
        
          with torch.no_grad():
        
               self.compute_advantage_values()
        
          self.update()
            
          return t
    
    
    def insert_data(self,obs,agent_actions,next_obs,rewards,dones):
        
        shared_obs = []
        
        num_stocks = self.NUM_STOCKS
        
        for i in range(self.num_agents):
            
            local_obs = []
            
            local_obs.append(obs[i].squeeze())
        
            for j in range(self.num_agents):
            
                if j != i:

                  local_obs.append(obs[j][0,1+num_stocks:1 + 2 * num_stocks])
                    
            local_obs = np.concatenate(local_obs).flatten()
           
            shared_obs.append(local_obs)
       
        with torch.no_grad():
            
             pred_val = []
        
             for i in range(self.num_agents):

                 val = self.critic_old[i](torch.FloatTensor(shared_obs[i]))
                
                 pred_val.append(val)
            
             old_logprob = []
         
             for i in range(self.num_agents):
                
                 prob,entropy = self.compute_logprob(obs[i],agent_actions[i],i)  
            
                 old_logprob.append(prob)
          
        
        for i in range(self.num_agents):
            
            self.replay_buffer[i].insert_data(obs[i],shared_obs[i],agent_actions[i],rewards[i],next_obs[i],dones[i],pred_val[i],old_logprob[i])
  
    
    def update(self):
        
        num_updates = int((self.replay_buffer[0].size / self.batchsize) * 10)
        
        for i in range(self.num_agents):
            
            for j in tqdm(range(num_updates)):       
            
                self.train_step(i)
            
    def train_step(self,agent_idx):
 
        indices = torch.randint(self.replay_buffer[agent_idx].size, size=(self.batchsize,), requires_grad=False).tolist()
                   
        reward_tensor = self.rewards[agent_idx]
        
        advantage_tensor = self.advantage[agent_idx]

        state = None

        action = None

        shared_state = None

        state = [self.replay_buffer[agent_idx].cur_obs_buffer[index] for index in indices]
        
        action = [self.replay_buffer[agent_idx].action_buffer[index] for index in indices]
                
        shared_state = [self.replay_buffer[agent_idx].shared_obs_buffer[index] for index in indices]
        
        new_action_logprob, dist_entropy = self.compute_logprob(state,action,agent_idx)
        
        old_action_logprob = [self.replay_buffer[agent_idx].logprob_buffer[index] for index in indices]
        
        old_action_logprob = torch.FloatTensor(old_action_logprob)
        
        ratio = torch.exp(new_action_logprob - old_action_logprob)
            
        weight = torch.clip(ratio,1-self.clip_param,1+self.clip_param)
                       
        advantage_clip = weight * advantage_tensor[indices]
    
        advantage_nonclip = ratio * advantage_tensor[indices]
    
        advantage_final = torch.min(advantage_clip,advantage_nonclip) 

        # actor update
           
        actor_loss = -1 * advantage_final.mean() + 0.02 * dist_entropy

        self.actor_optimizer[agent_idx].zero_grad() #reset gradients to 0
        
        actor_loss.backward(retain_graph=False) #this calculates the gradients
        
        self.actor_optimizer[agent_idx].step()   
    
    

        # critic update
                   
        q_val = self.critic[agent_idx](torch.FloatTensor(shared_state))
       
                               
        critic_loss = torch.nn.functional.mse_loss(reward_tensor[indices].squeeze(),q_val.squeeze())/(reward_tensor[indices].squeeze().std() + 1e-6)
        
        self.critic_optimizer[agent_idx].zero_grad() #reset gradients to 0
        
        critic_loss.backward(retain_graph=False) #this calculates the gradients
        
        self.critic_optimizer[agent_idx].step()      

        # network update

        tau = 2 ** -8
        
        self.net_update(self.critic[agent_idx],self.critic_old[agent_idx],tau)
        

    def calculate_action(self,agent_idx):
        
        state =  self.get_state(0,agent_idx)
        
        state = state.copy()

        state =  self.rescale_state(state)
        
        state = torch.from_numpy(state).float()
        
        mu, sigma = self.actor[agent_idx](state)
        
        cov_mat = torch.diag(sigma).unsqueeze(dim=0)
            
        dist = MultivariateNormal(mu, cov_mat)
        
        action = dist.sample()
           
        return action
      
    
    def step(self,obs,agent_idx):
        
        state = obs.copy()
        
        state = torch.from_numpy(state).float()
        
        mu, sigma = self.actor[agent_idx](state)
        
        cov_mat = torch.diag(sigma).unsqueeze(dim=0)
            
        dist = MultivariateNormal(mu, cov_mat)
        
        action = dist.sample()
        
        return action
    
    def compute_logprob(self,state,action,agent_idx):
        
        state1 = torch.FloatTensor(state)
              
        mu, sigma = self.actor[agent_idx](state1)
        
        action_var = sigma.expand_as(mu)
            
        cov_mat = torch.diag_embed(action_var)
            
        dist = MultivariateNormal(mu, cov_mat)
            
        action_logprobs = dist.log_prob(torch.FloatTensor(action))    
        
        dist_entropy = (action_logprobs.exp() * action_logprobs).mean()
            
        return action_logprobs,dist_entropy
    
    
    def compute_advantage_values(self):
        
        self.rewards.clear()
        
        self.advantage.clear()
        
        for i in range(self.num_agents):
        
            reward,advantage = self.compute_advantage_reward(i)
            
            self.rewards.append(reward)
            
            self.advantage.append(advantage)
            
    
    def compute_advantage_reward(self,agent_idx):
        
        rewards = self.replay_buffer[agent_idx].reward_buffer
        
        values = self.replay_buffer[agent_idx].pred_val_buffer

        done = self.replay_buffer[agent_idx].done_buffer
        
        capacity = self.replay_buffer[agent_idx].size
        
        last_val = self.critic_old[agent_idx](torch.FloatTensor(self.replay_buffer[agent_idx].shared_obs_buffer[capacity-1]))
        
        advantage = list()
        
        reward = list()
        
        gamma = 0.98
        
        gae_alpha = 0.95
            
        next_v = last_val
        
        next_adv = 0
        
        next_done = 0
            
        for j in reversed(range(len(rewards))):
                              
                              
                delta = rewards[j] + gamma * next_v * (1 - next_done) - values[j]
            
                adv_val = delta + gae_alpha * gamma * (1 - next_done) * next_adv
                
                reward.append(adv_val + values[j])
                
                next_adv = adv_val
                
                next_v = values[j]
                                                       
                next_done = done[j]                                      
                              
                advantage.append(adv_val)
            
                              
        advantage.reverse()   
            
        reward.reverse()
        
        reward_ten = torch.FloatTensor(reward)
    
        advantage_ten =  torch.FloatTensor(advantage)
        
        advantage_ten = (advantage_ten - advantage_ten.mean()) / (advantage_ten.std() + 1e-4)

                                      
        return reward_ten, advantage_ten         
           
        
    def eval(self):
    
        for agent_idx in range(self.num_agents):
        
            self.actor[agent_idx].eval()
        
            self.actor_old[agent_idx].eval()
        
            self.critic[agent_idx].eval()
        
            self.critic_old[agent_idx].eval()

    def train(self):
    
        for agent_idx in range(self.num_agents):
        
            self.actor[agent_idx].train()
        
            self.actor_old[agent_idx].train()
        
            self.critic[agent_idx].train()
        
            self.critic_old[agent_idx].train()
        
    def load_weights(self, output):
    
        for agent_idx in range(self.num_agents):
      
            self.actor[agent_idx].load_state_dict(torch.load("{}/actor{}.pkl".format(output,agent_idx)))
    
            self.actor_old[agent_idx].load_state_dict(
            
                 torch.load("{}/actor_old{}.pkl".format(output,agent_idx))
        
            )
        
            self.critic[agent_idx].load_state_dict(torch.load("{}/critic{}.pkl".format(output,agent_idx)))
        
            self.critic_old[agent_idx].load_state_dict(
            
            torch.load("{}/critic_old{}.pkl".format(output,agent_idx))
        
        )

    def save_model(self, output):
        
        for agent_idx in range(self.num_agents):
      
            torch.save(self.actor[agent_idx].state_dict(), "{}/actor{}.pkl".format(output,agent_idx))
        
            torch.save(self.critic[agent_idx].state_dict(), "{}/critic{}.pkl".format(output,agent_idx))
        
            torch.save(self.actor_old[agent_idx].state_dict(), "{}/actor_old{}.pkl".format(output,agent_idx))
        
            torch.save(self.critic_old[agent_idx].state_dict(), "{}/critic_old{}.pkl".format(output,agent_idx))
   
        
    
    