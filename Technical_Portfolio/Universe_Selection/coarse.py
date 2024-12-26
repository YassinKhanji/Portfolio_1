import sys  
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
# Add the directory containing data.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Management')))
from data import Data

class Coarse_1():
    def __init__(self):
        pass

    def universe(df):
        universe = []
        for coin in df.unstack().columns.levels[1]:
            if coin not in universe:
                universe.append(coin)

        return universe

    def volume_flag(self, df, max_dollar_allocation = 1000000):
        df['volume_flag'] =  np.where(df['volume_in_dollars'] * 0.05 > max_dollar_allocation, 1, 0)
        return df
   
    def sort_by_volume(self, df):
        """
        This function sorts the coins index by volume in descending order and keeps the top coins.

        It assumes that the input DataFrame has a MultiIndex with 'date' and 'coin' as the index levels.
        """


        """
        The reason why we removed the groupby is the same reason we removed for sort_by_std. 
        Since this might remove some data that may become essential later on when calculating returns, indicators, etc.
        we need to keep all the data and just add a column as a filter.
        """
        # df = (
        #     df.groupby(level='date', group_keys= False)  # Group by 'date'
        #     .apply(lambda group: group.sort_values('volume', ascending=False).head(top))  # Sort and take top 1
        # )

        # # Drop unused labels from the MultiIndex
        # df.index = df.index.remove_unused_levels()

        df['volume_rank'] = df['volume'].groupby(df.index.get_level_values(0)).rank(ascending = False)

        return df
    
    #Sort by standard deviation
    def sort_by_std(self, df, std_window, mean_window):
        """
        Calculates the normalized standard deviation (in %) of the returns for each coin
        and sorts the coins by standard deviation in descending order for each date.
        Keeps `std_values` for all coins in the final output.

        Parameters:
        - df: Copy of the DataFrame to store results.
        - std_window: Rolling window for standard deviation.
        - mean_window: Rolling window for mean.
        - top: Number of coins to keep with the highest standard deviation per date.

        Returns:
        - DataFrame with `std_values` for all coins and top `std_values` sorted per date.
        """
        #Unstack the data and df
        df_unstacked = df.copy().unstack()

        #Calculate the standard deviation
        for coin in df_unstacked.columns.get_level_values(1).unique():
            df_unstacked['std_values', coin] = df_unstacked['close', coin].rolling(window=std_window).std() / df_unstacked['close', coin].rolling(window = mean_window).mean()

        #Re-stack the df
        df = df_unstacked.stack(future_stack=True)

        #Filter by Standard Deviation
        """
        Originally, the code was written to filter the top coins with the highest standard deviation.

        What we realized is that missing many values inside make is more difficult to calculate returns and other indicators.
        So it is better to treat this as a filter and not a selection, thus making it a column in the DataFrame.

        Because we are not putting a threshold fo the standard deviation, we need to give each coin a rank based on the standard deviation.

        Then, when we are chosing which coin to keep, we can put a threshold to the rank (so top 10, top 20, etc.)
        """
        # df = (
        #     df.groupby(level='date', group_keys = False)  # Group by 'date'
        #     .apply(lambda group: group.sort_values('std_values', ascending=False).iloc[:top]))
        
        #  # Drop unused labels from the MultiIndex
        # #################### CALL THE BELOW EVERYTIME YOU WANT TO FILTER ####################
        # df.index = df.index.remove_unused_levels()

        df['std_rank'] = df['std_values'].groupby(df.index.get_level_values(0)).rank(ascending = False)

        """
        The following part is made to preserve the indicator data for all coins (filtered and non-filtered).
        Although we can skip it if we make a column for a filter (1 if filtered, 0 if not), like in fine filter,
        which is what we did.

        In reality, we don't need it since if we really need the indicator data, we can just calculate the values using 
            the original data. We don't need to do that for every function that filters and calculate indicator values.
        """
        # #unstack the df
        # df = df.unstack()

        # #Add the standard deviation data to the df
        # df['std_values'] = data_unstacked['std_values']

        # #Re-Stack the df
        # df = df.stack(future_stack = True)

        return df
        