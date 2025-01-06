import sys  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import pandas_ta as ta


# Ensure the directories are in the system path
sys.path.append(os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', 'Portfolio_1', 'Technical_Portfolio', 'Data_Management'))))
sys.path.append(os.path.abspath(os.path.abspath(os.path.join(os.getcwd(), '..', 'Portfolio_1', 'Technical_Portfolio', 'Universe_Selection'))))

# Import the modules
from data import Data
from coarse import Coarse_1 as Coarse
from fine import Fine_1 as Fine
from calculations import Calculations


class Trend_Following():
    
    def __init__(self):
        pass

    def supertrend_signals(self, df, length = 7, multiplier = 3):
        _df = df.copy().unstack()

        supertrend_results = {}
        # Iterate through each coin
        for coin in _df.columns.get_level_values(1).unique():  # Get unique coin names
            # Extract high, low, close for the coin
            high, low, close = _df["high", coin], _df["low", coin], _df["close", coin]

            # Calculate Supertrend
            supertrend = ta.supertrend(high, low, close, length, multiplier)

            supertrend_results[coin] = supertrend

        # Create a dataframe from the results
        supertrend_df = pd.concat(supertrend_results, axis=1)
        supertrend_df = supertrend_df.swaplevel(axis=1).sort_index(axis=1)

        _df = pd.concat([_df, supertrend_df], axis = 1)
        
        #Generate signals
        for coin in _df.columns.get_level_values(1):
            signal = (_df[f'SUPERTd_{length}_{float(multiplier)}', coin] == 1) & (_df[f'SUPERTd_{length}_{float(multiplier)}', coin].shift() == -1)
            _df['entry_signal', coin] = signal.astype(int)
            _df['entry_signal', coin] = _df['entry_signal', coin].shift().replace(np.nan, 0)

        # Stack the dataframe and get position and trades columns
        _df = _df.stack(future_stack=True)

        return _df
    
class Mean_Reversion():
        
    def __init__(self):
        pass

    def last_days_low(self, df, hourly_lookback = 1, daily_lookback = 1):
        """
        Assumes a stacked data that is hourly and has the columns: open, high, low, close

        Parameters:
        df: pd.DataFrame
        hourly_lookback: int
        daily_lookback: int
        """

        #Getting parameters
        start_time = df.index.levels[0][0].strftime('%Y-%m-%d')
        end_time = df.index.levels[0][-1].strftime('%Y-%m-%d')
        symbols = df.index.levels[1].unique()

        #Get the daily data and clean it
        df_daily = Data(symbols, '1d', start_time, end_time).df
        df_daily = df_daily[['open', 'high', 'low', 'close']]
        df_daily = df_daily.unstack().shift(daily_lookback).stack(future_stack = True)
        df_daily.columns = [f'shifted_daily_{col}' for col in df_daily.columns]

        #Reindex the daily data to match the hourly data
        df_daily_reindexed = df_daily.unstack().reindex(df[~df.index.duplicated()].unstack().index.get_level_values(0))\
        .ffill().stack(future_stack = True)


        #Concatenate the dataframes
        df = pd.concat([df, df_daily_reindexed], axis = 1)

        #Now to generate a direction column:
        # 1 if the close is above the daily close and last open is above the daily close and last close is below the daily close, else 0

        #Before that, we need to make sure we are dealing with the same date when comparing the daily low with the hourly closes
        _df = df[[]] #We don't need any of the columns, just the index (removing them to make sure it runs faster)
        _df.loc[:, 'date_only'] = _df.index.get_level_values(0).strftime('%Y-%m-%d') # Extract the date part from the datetime index
        _df.loc[:, 'previous_date'] = _df['date_only'].shift(1) # Shift the date column by one row
        _df['same_date'] = (_df['date_only'] == _df['previous_date']) # Compare the current date with the previous date
        df.loc[:, 'same_date']  = _df['same_date']
        df = df.unstack() #We have to unstack because we will be shifting columns later on for every single coin

        

        #Direction column
        for coin in df.columns.get_level_values(1).unique():
            df['last_days_low', coin] = df['same_date', coin] & (df['open', coin].shift(hourly_lookback) > df['shifted_daily_low', coin]) &\
            (df['close', coin].shift(hourly_lookback) < df['shifted_daily_low', coin]) & (df['close', coin] > df['shifted_daily_low', coin]) &\
            (df['close', coin].shift(hourly_lookback + 1) > df['shifted_daily_low', coin]) #Ensures that price is pulling back to the daily low, and not going from below it to above it

        df['entry_signal'] = df['last_days_low'].astype(int).shift(1) #We shift by one to avoid look ahead bias (we get the signals on the next candle)

        return df