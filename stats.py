import pandas as pd

from empyrical import sharpe_ratio,sortino_ratio,max_drawdown

from matplotlib import pyplot as plt


def calculate_statistics(total_asset_value):

    df_total_value = pd.DataFrame(total_asset_value)
        
    df_total_value = df_total_value.pct_change(1)

    sharpe = sharpe_ratio(df_total_value)

    sortino = sortino_ratio(df_total_value)

    mdd = max_drawdown(df_total_value) * 100

    return (sharpe,sortino,mdd)
    

def plot(reward_list,model_name,args):
        
    steps = [args.episode_length * i for i in range(len(reward_list))]

    plt.plot(steps,reward_list)

    plt.xlabel("steps")

    plt.ylabel("portfolio rewards")

    s = '%s_reward.png'%(model_name) 
  
    plt.savefig(s)

    plt.close()

def plot_portfolio_value(portfolio_values,date_range,model_name,args):

    color = "blue"

    plt.ylabel("Portfolio Value")

    plt.plot(date_range,portfolio_values,label=model_name,color = color)

    plt.legend(loc="upper left")

    plt.gcf().autofmt_xdate() # make space for and rotate the x-axis tick labels

    s = 'total_asset_value.png'
        
    plt.savefig(s)
    
    #plt.show()

def plot_multiple_portfolio_value(portfolio_values_list,date_range,model_list,color_list):

    fig, ax = plt.subplots()

    for i in range(len(portfolio_values_list)):

        plt.ylabel("Portfolio Value")

        print(len(portfolio_values_list[i]))

        print(date_range)

        print(model_list[i])
         
        print(color_list[i])

        plt.plot(date_range,portfolio_values_list[i],label=model_list[i],color = color_list[i])

        plt.legend(loc="upper left")

        plt.gcf().autofmt_xdate() # make space for and rotate the x-axis tick labels

        #s = 'total_asset_value_{}.png'.format(index)
        
        #        plt.savefig(s)


    ax.xaxis_date()     # interpret the x-axis values as dates

    fig.autofmt_xdate() # make space for and rotate the x-axis tick labels

    s = 'total_asset_value.png'
        
    plt.savefig(s)

    plt.show()