import requests
import json
import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from qgridnext import show_grid
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

class Stop_Loss():

    def __init__(self, df: pd.DataFrame, sl_type = 'atr', indicator_length = 0.0, sl_mult = 0.0, sl_percentage = 0.0, sl_dollar = 0.0):
        """
        Parameter:
            sl_type: string
                can be: supertrend, atr, dollar, percent
        Indicators currently implement: 
            Supertrend (for Dynamic SL)
            ATR (for fixed SL)
        """
        self.df = df
        self.indicator_length = indicator_length
        self.sl_mult = sl_mult
        self.sl_percentage = sl_percentage
        self.sl_dollar = sl_dollar
        self.sl_type = sl_type

    def define_sl_pos(self, group, coin):
        if (group['low', coin] <= group['session_stop_loss', coin]).any():
            start = group[group['low', coin] <= group['session_stop_loss', coin]].index[0]
            stop = group.index[-2]
            group.loc[start:stop, ("position", coin)] = 0
            return group
        else:
            return group
            
    def plot_sl(self, df):
        """
        We assume an unstacked dataframe with the following columns:
        - close
        - session_stop_loss
        """
        
        _df = df.copy()
        # Get unique coins
        unique_coins = _df.columns.get_level_values(1).unique()

        # Determine grid dimensions
        num_coins = len(unique_coins)
        cols = 5 # You can choose the number of columns
        rows = math.ceil(num_coins / cols)

        # Create the subplots grid
        fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(16, rows * 5))

        # Flatten the axes array for easy iteration
        axes = axes.flatten() #this makes the axes a 1D array, so we can iterate over it without going for another loop

        # Plot each coin
        for i, coin in enumerate(unique_coins):
            _df[[['close', coin], ['session_stop_loss', coin]]].plot(
                ax=axes[i],
                title=f'{coin} Close and Stop Loss',
                color=['blue', 'green'],
                legend = None
            )
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Closing Price')
            # axes[i].legend(title='Coin')

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def calculate_fixed_sl(self, plot = True):
        """
        This function applies a fixed stop loss to the position column in the dataframe.
        Returns a stacked DataFrame.

        Parameters:
            plot: boolean
                will plot the closing prices, position, as well as each of the session losses for each coin  
        """

        df = self.df.copy()

        #Calculate the ATR indicator
        if self.sl_type.lower() == 'atr':
            #unstack dataframe
            _df = df.copy().unstack()
            for coin in _df.columns.levels[1]:
                high, low, close = _df['high', coin], _df['low', coin], _df['close', coin]
                _df['atr', coin] = ta.atr(high, low, close, length=self.indicator_length)

            _df = _df.iloc[self.indicator_length:] #Slice the dataframe to remove the NaN values from the ATR calculation
            #It is better to do the above as we might get NaN values in other columns, so this might remove many needed rows
            #Note: This is done at this first stage right after we calculate all the indicators, We need to create a function in the 
                #future to remove the largest length needed for the calculations as this would be essential to warm up the data needed.
            
            #Calculate the stop loss
            _df = _df.stack(future_stack = True)
            _df['stop_loss'] = _df['close'] - self.sl_mult * _df['atr']

        elif self.sl_type.lower() == 'percent':
            _df['stop_loss'] = _df['close'] * (1 - self.sl_percentage)

        elif self.sl_type.lower() == 'dollar':
            _df['stop_loss'] = _df['close'] - self.sl_dollar


        ####Everything that comes after this is common for all fixed stop losses (percentage, dollar, indicator based, ...)#####

        #Unstack the dataframe
        _df = _df.unstack()

        #Calculate the session stop loss
        for coin in _df.columns.levels[1]:
            _df['session_stop_loss', coin] = _df['stop_loss', coin].groupby(_df['session', coin]).transform('first')
            # Group by both the session and coin, then pass the coin as an additional argument
            _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_sl_pos(group, coin))


        if plot:
            self.plot_sl(_df)

        #Stack the dataframe
        _df = _df.stack(future_stack=True)

        #Redefine the class parameter
        self.df = _df

        return _df
    
    #Note: we are going to use the same define_sl_pos() and plot_sl() as before

    def calculate_dynamic_sl(self, plot = True):
        """
        This function applies a dynamic stop loss to the position column in the dataframe.
        Returns a stacked DataFrame.

        Parameters:
            plot: boolean
                will plot the closing prices, position, as well as each of the session losses for each coin  
        """
        _df = self.df.copy().unstack()

        if self.sl_type.lower() == 'supertrend':
            if not any('SUPERT'.lower() in col.lower() for col in _df.columns.get_level_values(0)):
                #Calculate the supertrend indicator
                _df = Trend_Following().supertrend_signals(_df, self.indicator_length, self.sl_mult) #it contains supertrend values as well as signals
                #Slice the data to remove the warm up period of the indicator
                _df = _df.iloc[self.indicator_length:]
            
            #In all cases, rename the supertrend long column to be used as a stop loss
            for coin in _df.columns.levels[1]:
                _df['stop_loss', coin] = _df[f'SUPERTl_{self.indicator_length}_{float(self.sl_mult)}', coin]  

        elif self.sl_type.lower() == 'percent':
            _df['stop_loss'] = _df['close'] * (1 - self.sl_percentage)

        elif self.sl_type.lower() == 'dollar':
            _df['stop_loss'] = _df['close'] - self.sl_dollar

        for coin in _df.columns.levels[1]:
            _df['session_stop_loss', coin] = _df['stop_loss', coin].groupby(_df['session', coin]).transform(lambda x: x)

            # Group by both the session and coin, then pass the coin as an additional argument
            _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_sl_pos(group, coin))

        if plot:
            self.plot_sl(_df)

        _df = _df.stack(future_stack= True)

        return _df
        
    
    def apply_stop_loss(self, fixed = True, plot = True):
        """ 
        This function applies the fixed stop loss to the dataframe and calculates the trades, strategy returns, strategy cumulative returns, and sessions.
        it acts as a wrapper for the calculate_fixed_sl function, and make the necessary adjustments after that position column has changed

        Parameters:
            fixed: Boolean
                specifies if the stop loss will be fixed (True) or dynamic (False)
        """
        
        _df = self.calculate_fixed_sl(plot) if fixed else self.calculate_dynamic_sl(plot)
        _df = Calculations().trades(_df)
        _df = Calculations().strategy_returns(_df)
        _df = Calculations().strategy_creturns(_df)
        _df = Calculations().sessions(_df)
        return _df
    

    
            
        

    


    
    