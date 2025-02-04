import sys  
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
# Add the directory containing data.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Management')))
from data import Data

class Fine_1():
    def __init__(self):
        self.universe = []

    def universe(self, df):
        for coin in df.unstack().columns.levels[1]:
            if coin not in self.universe:
                self.universe.append(coin)

        return self.universe
    
    #Now Make it as a function
    def above_ema(self, df, ema_window, low_freq = '1d'):
        """
        parameters:
            data: pd.DataFrame (stacked with original data)
            df: pd.DataFrame (stacked)
            ema_window: int
        """
        df_1 = df.copy()
        htf_df = df_1.groupby(level = -1).resample(low_freq, level = 0).agg({
            'open': 'first',    # First value of the day
            'high': 'max',      # Maximum value of the day
            'low': 'min',       # Minimum value of the day
            'close': 'last',    # Last value of the day
            'volume': 'sum',     # Total volume of the day
            'volume_in_dollars': 'sum' # Total volume in dollars of the day
        })
        htf_df.columns = [f'fine_htf_{col}' for col in htf_df.columns]
        htf_df = htf_df.reorder_levels([1, 0], axis = 0).sort_index(axis = 0)
        htf_df = htf_df.unstack()
        for coin in htf_df.columns.get_level_values(1).unique():
            htf_df[f'ema_{ema_window}', coin] = ta.ema(htf_df['fine_htf_close', coin], length=ema_window)
            
        htf_df = htf_df.stack(future_stack = True)
        htf_df['above_ema'] = (htf_df['fine_htf_close'].fillna(0) > htf_df[f'ema_{ema_window}'].fillna(0)).astype(int) #We want the value of the previous day
        htf_df['above_ema'] = htf_df['above_ema'].shift()
        #Now we want to reindex to the original data
        df = df_1.join(htf_df, how = 'outer')
        df = df.unstack().ffill().stack(future_stack = True)
        
        # df = df.unstack()
        # df['above_ema'] = df['above_ema'].shift(periods = 1, freq = low_freq) #We want the value of the previous day
        # df = df.stack(future_stack= True)
        
        return df