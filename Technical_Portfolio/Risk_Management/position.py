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
from tail_risk import Stop_Loss, Take_Profit, Exit

class Position():
    def __init__(self, df, _min = 0, _max = np.inf):
        """
        parametes:
            df: Stacked dataframe
            _min: Minimum multiple position size allowed
            _max: Maximum multiple position size allowed
        
        Note: _min, _max are used to clip the position size to a certain range. Only for the cumulated_position method
        """
        self.df = df.copy()     
        self._min = _min
        self._max = _max       
            

    def custom_positon(self, series):
        """
        This will take a series where a condition is met and return a series with the position
        The series can have values of either 1 (when condition is met) or other values (when condition is not met)

        It is basically a copy of the series while keeping only the values of 1.

        Assumes a stacked column (if the series has a multiindex of multicoins)
        returns the position column
        """
        position = np.where(series == 1, 1, 0)
        position = pd.Series(position, index = series.index)
        position = position.shift(test.index.get_level_values(1).nunique()).fillna(0) 
        return position


    def calculate_position(self, df, _min = 0, _max = np.inf):
        """
        assumes a unstacked dataframe

        Calculates the 'Position' column, accumulating positions based on entry signals.

        Note that entry signals have a lower bound of 0 (without higher bound theoritacally), and exit signals have a range between 0 and 1

        Args:
            df: Pandas DataFrame with 'entry_signal' (0-1) and 'exit_signal' (boolean) columns.

        Returns:
            Pandas DataFrame with added 'Position' (float) and 'Session' (int) columns.
            Returns original dataframe if entry_signal and exit_signals columns are not found.
        """
        if _min < 0:
            raise ValueError("We can't take short positions, _min should be at least 0")
        
        if 'entry_signal' not in df.columns or 'exit_signal' not in df.columns:
            raise ValueError("Error: DataFrame must contain 'entry_signal' and 'exit_signal' columns.")
            
        for coin in df.columns.get_level_values(1).unique():
            df.loc[:, ('position', coin)] = 0.0

            for i in range(len(df)):
                if df.loc[df.index[i], ('entry_signal', coin)] > 0:
                    df.loc[df.index[i], ('position', coin)] = (df.loc[df.index[i-1], ('position', coin)] if i > 0 else 0) + df.loc[df.index[i], ('entry_signal', coin)]
                    # The above will add the entry to the current position

                elif df.loc[df.index[i], ('exit_signal', coin)] > 0:
                    if df.loc[df.index[i-1], ('position', coin)] > 0 if i > 0 else False: # check to see if we were in a position previously
                        current_position = df.loc[df.index[i-1], ('position', coin)]
                        new_position = current_position - current_position * df.loc[df.index[i], ('exit_signal', coin)]
                        df.loc[df.index[i], ('position', coin)] = new_position 

                elif df.loc[df.index[i-1], ('position', coin)] > 0 if i > 0 else False:
                    df.loc[df.index[i], ('position', coin)] = df.loc[df.index[i-1], ('position', coin)]

            df.loc[:, ('position', coin)] = df.loc[:, ('position', coin)].shift(1)
            df.loc[:, ('position', coin)] = np.clip(df.loc[:, ('position', coin)], _min, _max)
        
        return df
    
    def cumulated_position(self):
        """
        
        """
        _df = self.df.unstack()
        for coin in _df.columns.get_level_values(1):
            _df[f'position', coin ] =  _df['entry_signal', coin].cumsum().shift(1).fillna(0)
            
        _df = _df.stack(future_stack=True)
        #Perform some calculations
        _df = cal.trades(_df)
        _df = cal.strategy_returns(_df)
        _df = cal.sessions(_df)

        _df = self.calculate_position(_df, self._min, self._max)

        return _df.stack(future_stack=True)
