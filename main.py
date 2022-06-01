import pandas as pd
import argparse
import yaml
from Data_preprocess import DataDownloader
from Data_preprocess import Feature_Engineering
import itertools
from Data_preprocess import data_split
import torch
import os
from agents.agent import DDPGAgent
from agents.agent import PPOAgent
from agents.agent import TD3Agent
from multiagents.agent import MADDPGAgent
from multiagents.agent import MATD3Agent
from multiagents.agent import MAPPOAgent
from matplotlib import pyplot as plt
from tqdm.auto import tqdm
from stats import plot,plot_portfolio_value,plot_multiple_portfolio_value,calculate_statistics

MODELS = {"ddpg": DDPGAgent, "td3": TD3Agent, "ppo": PPOAgent, "maddpg": MADDPGAgent, "matd3": MATD3Agent, "mappo": MAPPOAgent}


def train(total_steps,agent,episode_length,output,args):
    
    t_step = 0
    
    agent.is_training = True
    
    best_sharpe = -9999
    
    best_reward = -99999
    
    step = 0
    
    num_epochs = int(total_steps/episode_length) + 1

    for _ in tqdm(range(num_epochs)):

          
          step = agent.run(episode_length,step)
            
          agent.is_training = False
            
          agent.eval()   

          args.is_test = False    
        
          stat,cur_reward,total_asset_value = agent.evaluate(args)

          agent.train()
            
          agent.is_training = True
       
          cur_sharpe = stat[0]        
        
          if cur_sharpe > best_sharpe:
               
             print("saving model!!!!")
                   
             best_sharpe = cur_sharpe
                   
             best_reward = cur_reward
                            
             agent.save_model(output)

    x_date = agent.env.df_eval.date.unique().tolist()

    date_range = pd.to_datetime(x_date) 
          
    plot_portfolio_value(total_asset_value,date_range,args.agent,args)  

    result = {
                  "sharpe": stat[0][0],
                  "sortino": stat[1],
                  "mdd": stat[2],
                  "return": (total_asset_value[len(total_asset_value)-1] - agent.env.budget)/agent.env.budget
             }

    print(result)
                             

def test(agent,output,args):

    agent.load_weights(output)
    
    agent.is_training = False
    
    agent.eval()

    args.is_test = True
    
    stat,reward,portfolio_value = agent.evaluate(args)

    result = {
                  "sharpe": stat[0][0],
                  "sortino": stat[1],
                  "mdd": stat[2],
                  "return": (portfolio_value[len(portfolio_value)-1] - agent.env.budget)/agent.env.budget
             }

    print(result)

    x_date = agent.env.df_test.date.unique().tolist()

    date_range = pd.to_datetime(x_date) 

    print(portfolio_value)

    for i in range(len(portfolio_value)):

        print(date_range[i],portfolio_value[i]) 

    plot_portfolio_value(portfolio_value,date_range,args.agent,args) 
    
           


if __name__ == "__main__":
    
   config_file = os.path.join(os.getcwd(),"config.yml")
    
   parser = argparse.ArgumentParser(description="PyTorch on stock trading using reinforcement learning")


   parser.add_argument(
      
       "--mode", default="train", type=str, help="option: train/test"
   )
    
   parser.add_argument(
       "--mid_dim1",
       default=64,
       type=int,
       help="hidden num of first fully connect layer of critic",
   )

   parser.add_argument(
       "--mid_dim2",
       default=64,
       type=int,
       help="hidden num of second fully connect layer of actor",
   )

   parser.add_argument("--batchsize", default = 2 ** 8, type=int, help="minibatch size")

   parser.add_argument(
      
       "--tau", default=2 **-8, type=float, help="moving average for target network"

   )

   parser.add_argument("--episode_length", default= 2 ** 12, type=int, help="")

   parser.add_argument(
      
       "--train_steps", default=200000, type=int, help =""

   )

   parser.add_argument(
      
       "--validate_episodes", default = 5, type=int, help =""

   )

   parser.add_argument(
      
       "--START_DATE", default = "2009-01-01", type=str, help =""

   )
   parser.add_argument(
      
       "--START_TRADE_DATE", default = "2020-06-30", type=str, help =""
      
   )
   parser.add_argument(
      
       "--END_TRADE_DATE", default = "2021-07-01", type=str, help =""
       
   )
   
   parser.add_argument(
     

       "--START_TEST_DATE", default = "2021-07-01", type=str, help =""

   )
    
   parser.add_argument(
      
       "--END_TEST_DATE", default = "2022-01-01", type=str, help =""

   )

   parser.add_argument(
      
       "--agent", default = "td3", type=str, help =""
   )
  
   parser.add_argument(
      
       "--num_agents", default = 3, type=str, help =""

   )

   parser.add_argument(
      
       "--is_test", default = False, type=str, help =""

   )

   parser.add_argument(
      
       "--is_eval", default = True, type=str, help =""

   )
   
   parser.add_argument(
      
       "--is_turbulence", default = False, type=str, help ="whether to use turbulence index for early selloff during bearish market"

   )
 
   args = parser.parse_args()

   with open(config_file, "r") as ymlfile:
    
        cfg = yaml.safe_load(ymlfile)
        
   tickers = cfg['DOW_12_TICKER']

   tech_indicators = cfg['TECHNICAL_INDICATORS_LIST']
 
   START_DATE = args.START_DATE

   START_TEST_DATE = args.START_TEST_DATE
  
   END_TEST_DATE = args.END_TEST_DATE

   START_TRADE_DATE = args.START_TRADE_DATE

   END_TRADE_DATE = args.END_TRADE_DATE


   df = DataDownloader( 
          start_date = START_DATE,
          end_date = END_TEST_DATE,
          stock_list = tickers,
        )

   df = df.fetch_data()

   feature = Feature_Engineering(tech_indicators)

   processed = feature.preprocess(df)   

   list_ticker = processed["tic"].unique().tolist()

   list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))

   combination = list(itertools.product(list_date,list_ticker))

   df_final = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")

   df_final = df_final[df_final['date'].isin(processed['date'])]

   df_final = df_final.sort_values(['date','tic'])

   train_df = data_split(df_final, START_DATE, START_TRADE_DATE)
    
   trade_df = data_split(df_final, START_TRADE_DATE, END_TRADE_DATE)

   test_df = data_split(df_final, START_TEST_DATE, END_TEST_DATE)

   state_dim = len(tickers) * 7 +1

   num_actions = len(tickers)

   mid_dim1 = args.mid_dim1

   mid_dim2 = args.mid_dim2

   args.output = os.path.join(os.getcwd(),"output")

   output = args.output
   
   if args.mode == "train":

       
        agent = MODELS[args.agent](state_dim,mid_dim1,mid_dim2,num_actions,train_df,trade_df,test_df,tech_indicators,args)

        train(args.train_steps,agent,args.episode_length,output,args)

      
            
   elif args.mode == "test":

        agent = MODELS[args.agent](state_dim,mid_dim1,mid_dim2,num_actions,train_df,trade_df,test_df,tech_indicators,args)
        
        test(agent,args.output,args)

   else:
        raise RuntimeError("undefined mode {}".format(args.mode))
        
   
     