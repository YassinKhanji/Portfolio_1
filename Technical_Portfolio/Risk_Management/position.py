import requests
import json
import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys  
import os
import pandas_ta as ta


# Ensure the directories are in the system path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'Data_Management')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'Universe_Selection')))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', 'Signal_Generation')))

# Import the modules
from data import Data
from calculations import Calculations
from coarse import Coarse_1 as Coarse
from fine import Fine_1 as Fine
from entry_signal import Trend_Following, Mean_Reversion
from tail_risk import Stop_Loss, Take_Profit  

class Position():
    def __init__(self, df):
        """
        parametes:
            df: Stacked dataframe
        """
        self.df = df.copy()

    def cumulated_position(self, df, min = 0, max = None):
        """
        This method is a combination of type of positions that wih a max and with a min. Typically, min = 0 and max = 1
        """
        signals_column = [col for col in self.df.columns if 'signals' in col]
        for coin in self.df.columns.get_level_values(1):
            self.df['position', coin] = self.df[''] 
        if max is not None:

            
            

    def custom_positon(self, series):
        """
        This will take a series where a condition is met and return a series with the position
        The series can have values of either 1 (when condition is met) or other values (when condition is not met)

        It is basically a copy of the series while keeping only the values of 1.

        Assumes a stacked column (if the series has a multiindex of multicoins)
        returns the position column
        """
        self.df['position'] = np.where(series == 1, 1, 0)
        # position = pd.DataFrame(position)
        self.df['position'] = self.df['position'].shift(len(self.df.index.get_level_values(1).unique())).fillna(0)

        _df = self.df
        
        return _df        

