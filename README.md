# STOCKS TRADING USING DEEP REINFORCEMENT LEARNING
STOCKS_RL currenly support the following model-free deep reinforcement learning algorithms:
1. DDPG, TD3, PPO  for continuous actions in single-agent environment
2. MADDPG, MATD3, MAPPO for continuous actions in multi-agent environment

# File Contents and Description

- agents
  - agent.py  #This file implement various single agent deep reinforcement learning algorithms
  - memory.py  #This file implements the replay buffer for various single agent deep reinforcement learning algorithms

- multiagents
  - agent.py  #This file implement various multi-agents deep reinforcement learning algorithms
  - memory.py  #This file implements the replay buffer for various multi-agents deep reinforcement learning algorithms

config.py -> This file contains all the tickers and technical indicators used for the stock market environment.

models.py -> This file contains implementations of the network architectures used in the deep reinforcement learning algorithms. 

environment.py -> This file contains the code for the stock trading market environment.

Data_preprocess.py -> This file contains the implementation for extracting stocks' historical pricing data and their associated turbulence value and technical indicators.

main.py -> This file kicks off the training of the model.

random_process.py -> This file contains implementation of random sampling for the DDPG algorithm.

stats.py -> This file contains implementations of sharpe ratio/max-drawdown/sortino ratio calculation of the portfolio.

# Installation Steps
Execute the following command to install the dependencies:
<code> pip install -r requirement.txt </code>

# Training
Execute the following command to train STOCKS_NEWS_RL:
<code>python main.py --train_steps 100000 --agent td3</code>


# Experimental Results
| 2020-07-01 - 2021-06-30 | DDPG | TD3 | PPO
| --- | --- | --- | --- |
| Initial Value| 1000000 | 1000000 | 1000000 |
| Final Value |1397352.4  |1417392.5  | 1452365.3|
| Sharpe Ratio | 2.28| 2.27 | 2.34 |
| Max DrawDown | -8.7 | -9.0 | -9.6 |



<img src="https://github.com/christopher-hma/STOCKS_TRADING_RL/blob/main/total_assets_value.png" width=150% height=100%>

 
