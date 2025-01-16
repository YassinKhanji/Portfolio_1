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
    def above_ema(self, df, ema_window):
        """
        parameters:
            data: pd.DataFrame (stacked with original data)
            df: pd.DataFrame (stacked)
            ema_window: int
        """
        df = df.copy().unstack()
        for coin in df.columns.get_level_values(1).unique():
            df[f'ema_{ema_window}', coin] = ta.ema(df['close', coin], length=ema_window)
        df = df.stack(future_stack = True)
        df['above_ema'] = (df['close'].fillna(0) > df[f'ema_{ema_window}'].fillna(0)).astype(int)
        return df