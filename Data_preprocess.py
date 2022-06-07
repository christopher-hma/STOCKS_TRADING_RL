import pandas as pd
import yfinance as yf
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf
import itertools

import scipy
import numpy as np
from scipy.spatial import distance
import linalg
from numpy.linalg import pinv

from matplotlib import pyplot as plt

class DataDownloader:
    
    def __init__(self, start_date: str, end_date: str, stock_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.stock_list = stock_list
        
    def format_time(self,df):
        
        df["date"] =  df["date"].dt.strftime("%Y-%m-%d")
        
        return df

    def fetch_data(self) -> pd.DataFrame:
       
        df = pd.DataFrame()
        for tic in self.stock_list:
            df_ = yf.download(tic, start=self.start_date, end=self.end_date)
            df_["tic"] = tic
            df = df.append(df_)
        # reset the index, we want to use numbers as index instead of dates
        df = df.reset_index()
        
        # convert the column names to standardized names
        df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjcp",
                "volume",
                "tic",
        ]
        # use adjusted close price instead of close price
        df["close"] = df["adjcp"]
        
        df = df.drop("adjcp", 1)
       
        # create day of the week column (monday = 0)
        df["day"] = df["date"].dt.dayofweek
        
        # convert date to standard string format, easy to filter
        df = self.format_time(df)
        
        # drop missing data
        df = df.dropna()
        df = df.reset_index(drop=True)
        print("Shape of DataFrame: ", df.shape)
        
        df = df.sort_values(by=['date','tic'])
        print(df)
        df = df.reset_index(drop=True)

        return df


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date", "tic"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


class Feature_Engineering:
   
    def __init__(self,technical_index):
        
        self.technical_index = technical_index
        

    def preprocess(self, df):
        
        df = self.clean_data(df)
        
        df = self.add_technical_indicator(df)
            
        df = df.fillna(method="bfill").fillna(method="ffill")
        
        return df
    
    def clean_data(self, data):
    
        df = data.copy()
        df = df.sort_values(['date','tic'],ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index = 'date',columns = 'tic', values = 'close')
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]
        
        return df

    def smooth(self,turbulence_data, half_life):
       
        import math
        
        half_life = float(half_life)
        
        smoothing_factor = 1 - math.exp(math.log(0.5) / half_life)
        
        smoothed_values = [turbulence_data[0]]
        
        for index in range(1, len(turbulence_data)):
            previous_smooth_value = smoothed_values[-1]
            
            new_unsmooth_value = turbulence_data[index]
            
            new_smooth_value = ((smoothing_factor * new_unsmooth_value)
                + ((1 - smoothing_factor) * previous_smooth_value))
            
            smoothed_values.append(new_smooth_value)
        
        return(smoothed_values)

    def gini(self, values):
        
        values = list(values)
        values.sort()
        
        minimum_value = values[0]
        lorenz_curve_value = minimum_value
        average_input = sum(values)/len(values)
        line_of_equality = [average_input]
        gap_area = [line_of_equality[0] - lorenz_curve_value]
        
        for index in range(1, len(values)):
            lorenz_curve_value += values[index]
            line_of_equality.append(line_of_equality[index - 1] + average_input)
            gap_area.append(line_of_equality[index - 1] + average_input
                            - lorenz_curve_value)
        
        return(sum(gap_area)/sum(line_of_equality))
    
    
    def calculate_systemic_risk(self,df):
        
        dates = df.date.unique()        
        df1 = df.copy()
        df_pivot = df1.pivot(index="date", columns="tic", values="close")        
        systemic_risk = [0] * 250       
        start = 250
        i = start
        while i < len(dates):

            cov_matrix = df_pivot.iloc[i - 250 : i + 1].cov()
            eigenvalues = np.sort(np.linalg.eig(cov_matrix)[0])
            systemic_risk.append(self.gini(values=eigenvalues))
            i += 1


        systemic_risk = pd.DataFrame(
            {"date": df_pivot.index, "systemic_risk": systemic_risk}
        )

        df = df.merge(systemic_risk, on="date")

        df = df.sort_values(["date", "tic"]).reset_index(drop=True)    

        print(df)
            
        return df
    
    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        
        df = df.sort_values(by=['tic','date'])
        
        stock = Sdf.retype(df.copy())
        
        print(stock)
        
        stock_names = stock.tic.unique()
                
        tech_indicator = self.technical_index
        
        for indicator in tech_indicator:
        
             df_indicator = pd.DataFrame()
        
             for i in range(len(stock_names)):
            
                 try:
                   
                     indicator_ = stock[stock.tic == stock_names[i]][indicator]
                     indicator_ = pd.DataFrame(indicator_)
                     indicator_['tic'] = stock_names[i]
                     size = len(df[df.tic == stock_names[i]]['date'].to_list())                
                     assert len(indicator_) == size
                     indicator_['date'] = df[df.tic == stock_names[i]]['date'].to_list()
                     df_indicator = df_indicator.append(
                           indicator_, ignore_index=True
                     )
                 except Exception as e:
                 
                     print(e)
                
             
             df = df.merge(df_indicator[['tic','date',indicator]],on=['tic','date'],how='left')
        
        df = df.sort_values(by=['date','tic'])
        
        df1 = df.copy()
    
        dates = df1.date.unique()
    
        start = 250
    
        i = start
    
        df_pivot = df1.pivot(index="date", columns="tic", values="close")
    
        turbulence_index = [0] * 250

        turbulence_sign = list()
 

        for i in range(start,len(dates)):

        
             historical_price = df_pivot.iloc[i-start:i]
        
             historical_price_mean = historical_price.mean()

             current_price = df_pivot.iloc[[i]]

             historical_price_cov = pinv(historical_price.cov())
   
             dist = distance.mahalanobis(current_price, historical_price_mean, historical_price_cov)**2
        
             turbulence_index.append(dist)
        
             i += 1

        for i in range(250):

            turbulence_sign.append(1)

        for i in range(250,len(dates)):

           
            ratio = (turbulence_index[i] - turbulence_index[i - 15])/turbulence_index[i - 15]


            if abs(ratio) >= 0.33:

               print(dates[i],turbulence_index[i],turbulence_index[i - 15])

               turbulence_sign.append(-1)

               #print(dates[i],turbulence_index[i],turbulence_index[i - 15])

            else:

               turbulence_sign.append(1)

       
        turbulence_sign = pd.DataFrame(
            {"date": df_pivot.index, "turbulence_sign": turbulence_sign}
        )

        df = df.merge(turbulence_sign, on="date")
        
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)    

        #turbulence_index = self.smooth(turbulence_index,12)        

        turbulence_index = pd.DataFrame(
            {"date": df_pivot.index, "turbulence": turbulence_index}
        )

   
        df = df.merge(turbulence_index, on="date")
        
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)

        return df