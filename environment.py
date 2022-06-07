import math
import torch.nn.functional as F
import copy
import torch.nn as nn
import numpy as np
import pandas as pd
import copy


import math
import torch.nn.functional as F
import copy
import torch.nn as nn
import numpy as np
import pandas as pd
import copy


class StockTradeEnvironment():
    
    def __init__(self,num_actions,df,df_eval,df_test,tech,is_turbulence,turbulence_threshold = 27.1):
        
        super(StockTradeEnvironment, self).__init__()
        
        self.df = df
        
        self.df_eval = df_eval
        
        self.df_test = df_test
        
        self.NUM_STOCKS = num_actions
        
        self.budget = 1000000
        
        self.portfolio_state = np.zeros([1,(self.NUM_STOCKS * (len(tech) + 2) + 1)])      
        
        self.eval_portfolio_state = np.zeros([1,(self.NUM_STOCKS * (len(tech) + 2) + 1)])      
        
        self.test_portfolio_state = np.zeros([1,(self.NUM_STOCKS * (len(tech) + 2) + 1)])
               
        self.datelist = df.index.unique().tolist()        
        
        self.eval_datelist = df_eval.index.unique().tolist()
        
        self.test_datelist = df_test.index.unique().tolist()
        
        print(self.eval_datelist)

        self.tech = tech

        self.eval_initial_state(self.eval_datelist[0])
        
        self.test_initial_state(self.test_datelist[0])
       
        self.initial_state(self.datelist[0])
                
        self.maxShare = 1e2
        
        self.turbulence_threshold = turbulence_threshold
         
        self.turbulence = 0

        self.is_turbulence = is_turbulence
        
        self.systemic_risk = 0
        
        self.risk_trend_up = 0
        
        self.systemic_risk_old = 0
    
    def initial_state(self,date):
        
        
        self.portfolio_state[0,0] = self.budget
        
        self.portfolio_state[0,1:1+self.NUM_STOCKS] = self.df.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
            self.portfolio_state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df.loc[date][tech].values
         
        self.turbulence = self.df.loc[0]["turbulence_sign"].values[0]

        self.systemic_risk = self.df.loc[0]["systemic_risk"].values[0]
        
        self.systemic_risk_old = 0
        
        self.risk_trend_up = 0
        
    def eval_initial_state(self,date):

        self.eval_portfolio_state[0,0] = self.budget
        
        self.eval_portfolio_state[0,1:1+self.NUM_STOCKS] = self.df_eval.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
            self.eval_portfolio_state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df_eval.loc[date][tech].values
        
        self.turbulence = self.df_eval.loc[0]["turbulence_sign"].values[0]
            
        self.systemic_risk = self.df_eval.loc[0]["systemic_risk"].values[0]
        
        self.systemic_risk_old = 0       
        
        self.risk_trend_up = 0
        
    def test_initial_state(self,date):
        
        self.test_portfolio_state[0,0] = self.budget
        
        self.test_portfolio_state[0,1:1+self.NUM_STOCKS] = self.df_test.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
            self.test_portfolio_state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df_test.loc[date][tech].values
        
        self.turbulence = self.df_test.loc[0]["turbulence_sign"].values[0]
            
        self.systemic_risk = self.df_test.loc[0]["systemic_risk"].values[0]
        
        self.systemic_risk_old = 0
        
        self.risk_trend_up = 0
        
    def risk_increase(self):
        
        return self.risk_trend_up > 3
        
    def sell_stock(self,sell_index,is_eval,is_test):
        
        if (self.is_turbulence and self.turbulence == -1) or (self.risk_increase() and self.is_turbulence):
 
           state = self.get_state(is_eval,is_test)
        
           share_sell_quantity = state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS].copy()
            
           state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] -= share_sell_quantity
        
           state[0,0] += np.sum(share_sell_quantity * state[0,1 : 1 + self.NUM_STOCKS])

           
        else:
        
           state = self.get_state(is_eval,is_test)
        
           share_sell_quantity = np.minimum(-sell_index, state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS])

           state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] -= share_sell_quantity
        
           state[0,0] += np.sum(share_sell_quantity * state[0,1 : 1 + self.NUM_STOCKS])
   

    def buy_stock(self,buy_index,is_eval,is_test):
        
        state = self.get_state(is_eval,is_test)
        
        if (self.is_turbulence and self.turbulence == -1) or (self.risk_increase() and self.is_turbulence):
       
            pass
        
        else:
            
            for i in range(len(buy_index)):
            
                if(state[0,0] / state[0,1+i] >= buy_index[i]):
                
                    state[0,0] -= state[0,1+i] * buy_index[i]
                
                    state[0,1+self.NUM_STOCKS+i] += buy_index[i]
             
            
                else:
                
                    buy_quantity = math.floor(state[0,0] / state[0,1+i])
                
                    state[0,0] -= state[0,1+i] * buy_quantity
                
                    state[0,1+self.NUM_STOCKS+i] += buy_quantity
                
    def step(self,actions,t,T,is_eval=False,is_test=False):

        prev_state = self.get_state(is_eval,is_test).copy()
 
        reward = None

        if self.systemic_risk > self.systemic_risk_old: 
        
           self.risk_trend_up += 1
            
        else:
            
           self.risk_trend_up = 0
        
        self.systemic_risk_old = self.systemic_risk
        
        self.make_transaction(actions,is_eval,is_test)
 
        self.update(t+1,is_eval,is_test)

        reward = self.CalculateReward(prev_state,is_eval,is_test)

        done = 1 if t == T - 2 else 0 

        next_state = self.get_state(is_eval,is_test).copy() if ( t < T - 2 or is_eval or is_test) else self.reset_()

        next_time = 0 if t == T - 2 else t+1 

        return next_state,reward,done,next_time

            
    def make_transaction(self,actions,is_eval,is_test):
               
        actions = actions * self.maxShare
        
        sell_index = copy.deepcopy(actions)
        
        buy_index = copy.deepcopy(actions)
    
        
        buy_index[buy_index < 0] = 0
        
        sell_index[sell_index > 0] = 0
        
        
        self.sell_stock(sell_index.astype(int),is_eval,is_test)
        
        self.buy_stock(buy_index.astype(int),is_eval,is_test)
        
    
    def CalculateReward(self,prev_State,is_eval,is_test):

        state = self.get_state(is_eval,is_test)

        new_value = state[0,0] + np.sum(state[0,1:1+self.NUM_STOCKS] * state[0,1+self.NUM_STOCKS:1+ 2 * self.NUM_STOCKS])        

        old_value = prev_State[0,0] + np.sum(prev_State[0,1:1+self.NUM_STOCKS] * prev_State[0,1+self.NUM_STOCKS:1+ 2 * self.NUM_STOCKS])
        
        return new_value - old_value
    
    def update(self,date,is_eval,is_test):
        
        state = self.test_portfolio_state if is_test else (self.eval_portfolio_state if is_eval else self.portfolio_state)

        df = self.df_test if is_test else (self.df_eval if is_eval else self.df)

        date = self.datelist[date]
        
        state[0,1:1+self.NUM_STOCKS] = df.loc[date]["close"].values

        for i,tech in enumerate(self.tech):
        
              state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = df.loc[date][tech].values
            
        self.turbulence = df.loc[date]["turbulence_sign"].values[0]
        
        self.systemic_risk = df.loc[date]["systemic_risk"].values[0]

    def reset_(self):
        
        self.reset()
            
        self.initial_state(0)
        
        next_state = self.get_state(is_eval = False,is_test = False).copy()
            
        return next_state

    def eval_reset_(self):
        
        self.eval_reset()
            
        self.eval_initial_state(0)
        
        next_state = self.get_state(is_eval = True,is_test = False).copy()
                     
        return next_state
    
    def test_reset_(self):
        
        self.test_reset()
            
        self.test_initial_state(0)
        
        next_state = self.get_state(is_eval = False,is_test = True).copy()
                     
        return next_state
    
    def reset(self):

        self.portfolio_state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0
        
    def eval_reset(self):

        self.eval_portfolio_state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0
        
    def test_reset(self):

        self.test_portfolio_state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0
                            
    def get_state(self,is_eval,is_test):
 
        return self.test_portfolio_state if is_test else (self.eval_portfolio_state if is_eval else self.portfolio_state)


class MultiStockTradeEnvironment():
    
    def __init__(self,num_actions,num_agents,df,df_eval,df_test,tech,is_turbulence,turbulence_threshold = 27.1):
        
        super(MultiStockTradeEnvironment, self).__init__()
        
        self.df = df
        
        self.df_eval = df_eval
        
        self.df_test = df_test
        
        self.NUM_STOCKS = num_actions
        
        self.num_agents = num_agents
        
        self.budget = 1000000
        
        self.portfolio_state = [np.zeros([1,(self.NUM_STOCKS * (len(tech) + 2) + 1)]) for i in range(self.num_agents)]        
        
        self.eval_portfolio_state = [np.zeros([1,(self.NUM_STOCKS * (len(tech) + 2) + 1)]) for i in range(self.num_agents)]        
        
        self.test_portfolio_state = [np.zeros([1,(self.NUM_STOCKS * (len(tech) + 2) + 1)]) for i in range(self.num_agents)]
        
       
        self.datelist = df.index.unique().tolist()        
        
        self.eval_datelist = df_eval.index.unique().tolist()
        
        self.test_datelist = df_test.index.unique().tolist()
        
        self.tech = tech

        self.eval_initial_state(self.eval_datelist[0])
        
        self.test_initial_state(self.test_datelist[0])
       
        self.initial_state(self.datelist[0])
                
        self.maxShare = 1e2
         
        self.turbulence = 0
        
        self.is_turbulence = is_turbulence

        self.turbulence_threshold = turbulence_threshold
        
        self.systemic_risk = 0
        
        self.risk_trend_up = 0
        
        self.systemic_risk_old = 0
        

    def initial_state(self,date):
        
        
        for agent_idx in range(self.num_agents):
               
            self.portfolio_state[agent_idx][0,0] = self.budget
        
            self.portfolio_state[agent_idx][0,1:1+self.NUM_STOCKS] = self.df.loc[date]["close"].values
        
            for i,tech in enumerate(self.tech):
        
                self.portfolio_state[agent_idx][0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df.loc[date][tech].values
        
        self.turbulence = self.df.loc[0]["turbulence_sign"].values[0]
        
        self.systemic_risk = self.df.loc[0]["systemic_risk"].values[0]
        
        self.systemic_risk_old = 0
        
        self.risk_trend_up = 0

        
    def eval_initial_state(self,date):

        
        for agent_idx in range(self.num_agents):
                
            self.eval_portfolio_state[agent_idx][0,0] = self.budget
        
            self.eval_portfolio_state[agent_idx][0,1:1+self.NUM_STOCKS] = self.df_eval.loc[date]["close"].values
        
            for i,tech in enumerate(self.tech):
        
                self.eval_portfolio_state[agent_idx][0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df_eval.loc[date][tech].values
        
        self.turbulence = self.df_eval.loc[0]["turbulence_sign"].values[0]
        
        self.systemic_risk = self.df_eval.loc[0]["systemic_risk"].values[0]
        
        self.systemic_risk_old = 0
        
        self.risk_trend_up = 0
        
        
    def test_initial_state(self,date):
        
        for agent_idx in range(self.num_agents):
                
            self.test_portfolio_state[agent_idx][0,0] = self.budget
        
            self.test_portfolio_state[agent_idx][0,1:1+self.NUM_STOCKS] = self.df_test.loc[date]["close"].values
       
            for i,tech in enumerate(self.tech):
        
                self.test_portfolio_state[agent_idx][0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = self.df_test.loc[date][tech].values
       
        self.turbulence = self.df_test.loc[0]["turbulence_sign"].values[0]
        
        self.systemic_risk = self.df_test.loc[0]["systemic_risk"].values[0]
        
        self.systemic_risk_old = 0
        
        self.risk_trend_up = 0
                          
            
    def risk_increase(self):
        
        return self.risk_trend_up > 3
    
    def sell_stock(self,sell_index,agent_idx,is_eval,is_test):
        
        if (self.is_turbulence and self.turbulence == -1) or (self.risk_increase() and self.is_turbulence):
            
            state = self.get_state(agent_idx,is_eval,is_test)
            
            share_sell_quantity = state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS].copy()
            
            state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] -= share_sell_quantity
        
            state[0,0] += np.sum(share_sell_quantity * state[0,1 : 1 + self.NUM_STOCKS])
        
        else:

            state = self.get_state(agent_idx,is_eval,is_test)
        
            share_sell_quantity = np.minimum(-sell_index, state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS])

            state[0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] -= share_sell_quantity
        
            state[0,0] += np.sum(share_sell_quantity * state[0,1 : 1 + self.NUM_STOCKS])

    def buy_stock(self,buy_index,agent_idx,is_eval,is_test):
        
        if (self.is_turbulence and self.turbulence == -1) or (self.risk_increase() and self.is_turbulence):
                
            pass
       
        state = self.get_state(agent_idx,is_eval,is_test)

        for i in range(len(buy_index)):
            
            if(state[0,0] / state[0,1+i] >= buy_index[i]):
                
                state[0,0] -= state[0,1+i] * buy_index[i]
                
                state[0,1+self.NUM_STOCKS+i] += buy_index[i]
             
            
            else:
                
                buy_quantity = math.floor(state[0,0] / state[0,1+i])
                
                state[0,0] -= state[0,1+i] * buy_quantity
                
                state[0,1+self.NUM_STOCKS+i] += buy_quantity
                                
    def step(self,action_list,t,T,is_eval=False,is_test=False):

        next_obs = []
        
        dones = []
        
        rewards = []
        
        if self.systemic_risk > self.systemic_risk_old:
        
           self.risk_trend_up += 1
            
        else:
            
           self.risk_trend_up = 0
        
        self.systemic_risk_old = self.systemic_risk
        
        for agent_idx in range(len(action_list)):
            
            actions = action_list[agent_idx]

            prev_state = self.get_state(agent_idx,is_eval,is_test).copy()
 
            self.make_transaction(actions,agent_idx,is_eval,is_test)
 
            self.update(t+1,agent_idx,is_eval,is_test)

            reward = self.CalculateReward(prev_state,agent_idx,is_eval,is_test)

            dones.append(0)
            
            rewards.append(reward)
            
        df = self.df_test if is_test else (self.df_eval if is_eval else self.df)

        if ( t < T - 2 or is_eval or is_test):

            for agent_idx in range(len(action_list)):
 
                next_state = self.get_state(agent_idx,is_eval,is_test).copy()

                next_obs.append(next_state)
        
            self.turbulence = df.loc[t+1]["turbulence_sign"].values[0]
 
            self.systemic_risk = df.loc[t+1]["systemic_risk"].values[0]

        else:

            next_obs = self.reset_()

            self.turbulence = df.loc[0]["turbulence_sign"].values[0]
            
            self.systemic_risk = df.loc[0]["systemic_risk"].values[0]

        next_time = 0 if t == T - 2 else t+1 
    
            
        return next_obs,rewards,dones,next_time

            
    def make_transaction(self,actions,agent_idx,is_eval,is_test):
               
        actions = actions * self.maxShare
        
        sell_index = copy.deepcopy(actions)
        
        buy_index = copy.deepcopy(actions)   
        
        buy_index[buy_index < 0] = 0
        
        sell_index[sell_index > 0] = 0
        
        
        self.sell_stock(sell_index.astype(int),agent_idx,is_eval,is_test)
        
        self.buy_stock(buy_index.astype(int),agent_idx,is_eval,is_test)

    
    def CalculateReward(self,prev_State,agent_idx,is_eval,is_test):

        state = self.get_state(agent_idx,is_eval,is_test)
       
        new_value = state[0,0] + np.sum(state[0,1:1+self.NUM_STOCKS] * state[0,1+self.NUM_STOCKS:1+ 2 * self.NUM_STOCKS])
        
        old_value = prev_State[0,0] + np.sum(prev_State[0,1:1+self.NUM_STOCKS] * prev_State[0,1+self.NUM_STOCKS:1+ 2 * self.NUM_STOCKS])
        
        return new_value - old_value
    
    def update(self,date,agent_idx,is_eval,is_test):
        
        state = self.get_state(agent_idx,is_eval,is_test)
       
        df = self.df_test if is_test else (self.df_eval if is_eval else self.df)

        date = self.datelist[date]
        
        state[0,1:1+self.NUM_STOCKS] = df.loc[date]["close"].values
        
        for i,tech in enumerate(self.tech):
        
              state[0,1 + (i+2) * self.NUM_STOCKS: 1 + (i+3) *self.NUM_STOCKS] = df.loc[date][tech].values


    def reset_(self):

        self.initial_state(0)

        obs_list = []

        for agent_idx in range(self.num_agents):
        
            self.reset(agent_idx)
            
            next_state = self.get_state(agent_idx,is_eval = False,is_test = False).copy()

            obs_list.append(next_state)
            
        return obs_list

    def eval_reset_(self):
        
            
        self.eval_initial_state(0)

        obs_list = []
        
        for agent_idx in range(self.num_agents):

            self.eval_reset(agent_idx)

            next_state = self.get_state(agent_idx,is_eval = True,is_test = False).copy()

            obs_list.append(next_state)
                     
        return obs_list

    
    def test_reset_(self):
            
        self.test_initial_state(0)

        obs_list = []

        for agent_idx in range(self.num_agents):

            self.test_reset(agent_idx)
        
            next_state = self.get_state(agent_idx,is_eval = False,is_test = True).copy()

            obs_list.append(next_state)
                     
        return obs_list

        
    def reset(self,agent_idx):

        self.portfolio_state[agent_idx][0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0

    def eval_reset(self,agent_idx):

        self.eval_portfolio_state[agent_idx][0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0
        
    def test_reset(self,agent_idx):

        self.test_portfolio_state[agent_idx][0,1 + self.NUM_STOCKS : 1 + 2 * self.NUM_STOCKS] = 0
                            
    def get_state(self,agent_idx,is_eval,is_test):
 
        return self.test_portfolio_state[agent_idx] if is_test else (self.eval_portfolio_state[agent_idx] if is_eval else self.portfolio_state[agent_idx])

