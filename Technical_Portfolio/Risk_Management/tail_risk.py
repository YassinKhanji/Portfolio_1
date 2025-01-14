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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Management')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Universe_Selection')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Signal_Generation')))

# Import the modules
from data import Data
from calculations import Calculations
from coarse import Coarse_1 as Coarse
from fine import Fine_1 as Fine
from entry_signal import Trend_Following, Mean_Reversion

class Stop_Loss():

    def __init__(self, df: pd.DataFrame,
                 sl_type = 'atr', 
                 sl_ind_length = 0.0, 
                 sl_mult = 0.0, 
                 sl_percentage = 0.0, 
                 sl_dollar = 0.0,
                 exit_percent = 1.0, 
                 signal_only = True):
        """
        Parameter:
            sl_type: string
                can be: supertrend, atr, dollar, percent
            
            exit_amount: float
                amount to subtract from position (represents percentage to be sold when stop loss is executed).
                useful for partial stop losses.

            signal_only: boolean
                if True, the stop loss will only be a signal and not a position change

        Indicators currently implemented: 
            Supertrend (for Dynamic SL)
            ATR (for fixed SL)
        """
        self.df = df
        self.sl_ind_length = sl_ind_length
        self.sl_mult = sl_mult
        self.sl_percentage = sl_percentage
        self.sl_dollar = sl_dollar
        self.sl_type = sl_type
        self.exit_percent = exit_percent
        self.signal_only = signal_only

    def define_sl_pos(self, group, coin):
        current_pos = group['position', coin].iloc[-1]
        if (group['low', coin] <= group['session_stop_loss', coin]).any():
            start = group[group['low', coin] <= group['session_stop_loss', coin]].index[0]
            stop = group.index[-2]
            group.loc[start:stop, ("position", coin)] = current_pos - current_pos * self.exit_percent
            return group
        else:
            return group
    
    def define_sl_signal(self, group, coin):
        """
        We are looking for a high that is above the current take profit (of this session)
        """
        group[("exit_signal_sl", coin)] = 0  # Initialize with 0
        if (group['low', coin] <= group['session_stop_loss', coin]).any():
            start = group[group['low', coin] <= group['session_stop_loss', coin]].index[0]
            group.loc[start, ("exit_signal_sl", coin)] = self.exit_percent
            group.loc[start, ('price', coin)] = group.loc[start, ('session_stop_loss', coin)]
        return group
            
    def plot_sl(self, df):
        """
        Takes an unstacked dataframe with the following columns:
        - close
        - session_stop_loss
        """
        
        _df = df.copy().unstack()
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
            ax = _df[[['low', coin], ['session_stop_loss', coin]]].plot(
                ax=axes[i],
                title=f'{coin} Close and Stop Loss',
                color=['blue', 'green'],
                legend = None
            )
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Closing Price')
            # axes[i].legend(title='Coin')
            ax2 = ax.twinx()
            _df['position', coin].plot(ax = ax2, alpha = 0.5)

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def calculate_fixed_sl(self):
        """
        This function applies a fixed stop loss to the position column in the dataframe.
        Returns a stacked DataFrame.

        Parameters:
            plot: boolean
                will plot the closing prices, position, as well as each of the session losses for each coin  
        """

        _df = self.df.copy()
        print(f'length when calling: {len(_df)}')
        
        #Calculate the ATR indicator
        if self.sl_type.lower() == 'supertrend':
            raise ValueError("Fixed stop loss does not currently support the supertrend indicator, use Dynamic Stop Loss")
        elif self.sl_type.lower() == 'atr':
            #unstack dataframe
            print(f'length before unstacking: {len(_df)}')
            _df = _df.copy().unstack()
            print(f'length after unstacking: {len(_df)}')
            
            if not any('atr'.lower() in col.lower() for col in _df.columns.get_level_values(0)):
                #Calculate the ATR indicator
                for coin in _df.columns.levels[1]:
                    print(len(_df))
                    high, low, close = _df['high', coin], _df['low', coin], _df['close', coin]
                    _df['atr', coin] = ta.atr(high, low, close, length=self.sl_ind_length)
                    print(_df['atr', coin].value_counts())
                    print(len(_df))

                print(len(_df))
                print(self.sl_ind_length)
                _df = _df.iloc[self.sl_ind_length:] #Slice the dataframe to remove the NaN values from the ATR calculation
                #It is better to do the above as we might get NaN values in other columns, so this might remove many needed rows
                #Note: This is done at this first stage right after we calculate all the indicators, We need to create a function in the 
                    #future to remove the largest length needed for the calculations as this would be essential to warm up the data needed.
                print(len(_df))
                
            #Calculate the stop loss
            print(len(_df))
            _df = _df.stack(future_stack = True)
            _df['stop_loss'] = _df['close'] - self.sl_mult * _df['atr']
            print(len(_df))
            
        elif self.sl_type.lower() == 'percent':
            _df['stop_loss'] = _df['close'] * (1 - self.sl_percentage)

        elif self.sl_type.lower() == 'dollar':
            _df['stop_loss'] = _df['close'] - self.sl_dollar


        ####Everything that comes after this is common for all fixed stop losses (percentage, dollar, indicator based, ...)#####

        #Unstack the dataframe
        print(_df)
        _df = _df.unstack()
        print(_df)
        # _df = _df.fillna(0)
        #Calculate the session stop loss
        for coin in _df.columns.levels[1]:
            print(_df)
            _df['session_stop_loss', coin] = _df['stop_loss', coin].groupby(_df['session', coin]).transform('first')
            # Group by both the session and coin, then pass the coin as an additional argument
            if self.signal_only:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_sl_signal(group, coin))
            else:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_sl_pos(group, coin))


        #Stack the dataframe
        _df = _df.stack(future_stack=True)

        #Redefine the class parameter
        self.df = _df

        return _df
    
    #Note: we are going to use the same define_sl_pos() and plot_sl() as before

    def calculate_dynamic_sl(self):
        """
        This function applies a dynamic stop loss to the position column in the dataframe.
        Returns a stacked DataFrame.

        Parameters:
            plot: boolean
                will plot the closing prices, position, as well as each of the session losses for each coin  
        """

        _df = self.df.copy()
        if self.sl_type.lower() == 'atr':
            raise ValueError('Dynamic stop loss does not currently support the ATR indicator, use Fixed Stop Loss')
            
        elif self.sl_type.lower() == 'supertrend':
            _df = _df.unstack().copy()
            if not any('SUPERT'.lower() in col.lower() for col in _df.columns.get_level_values(0)):
                #Calculate the supertrend indicator
                _df = Trend_Following().supertrend_signals(_df, self.sl_ind_length, self.sl_mult) #it contains supertrend values as well as signals
                #Slice the data to remove the warm up period of the indicator
                _df = _df.iloc[self.sl_ind_length:]
            
            #In all cases, rename the supertrend long column to be used as a stop loss
            for coin in _df.columns.levels[1]:
                _df['stop_loss', coin] = _df[f'SUPERTl_{self.sl_ind_length}_{float(self.sl_mult)}', coin] 

            _df = _df.stack(future_stack = True) 

        elif self.sl_type.lower() == 'percent':
            _df['stop_loss'] = _df['close'] * (1 - self.sl_percentage)

        elif self.sl_type.lower() == 'dollar':
            _df['stop_loss'] = _df['close'] - self.sl_dollar


        _df = _df.unstack().copy()
        for coin in _df.columns.levels[1]:
            _df['session_stop_loss', coin] = _df['stop_loss', coin].groupby(_df['session', coin]).transform(lambda x: x)
            # Group by both the session and coin, then pass the coin as an additional argument
            if self.signal_only:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_sl_signal(group, coin))
            else:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_sl_pos(group, coin))


        _df = _df.stack(future_stack= True)

        return _df
        
    
    def apply_stop_loss(self, fixed = True, plot = True):
        """ 
        This function applies the stop loss to the dataframe and calculates the trades, strategy returns, strategy cumulative returns, and sessions.
        it acts as a wrapper for the calculate_fixed_sl function, and make the necessary adjustments after that position column has changed

        Parameters:
            fixed: Boolean
                specifies if the stop loss will be fixed (True) or dynamic (False).
                This is essential if we are using sl_type of percent or dollar.
        """
        
        _df = self.calculate_fixed_sl() if fixed else self.calculate_dynamic_sl()
        if plot:
            self.plot_sl(_df)
        _df = Calculations().trades(_df)
        _df = Calculations().strategy_returns(_df)
        _df = Calculations().strategy_creturns(_df)
        _df = Calculations().sessions(_df)
        return _df
    

class Take_Profit():

    def __init__(self, 
                 df, 
                 tp_type = 'rr', 
                 tp_mult = 2.0, 
                 tp_ind_length = 0,
                 tp_percent = 0.02,
                 tp_dollar = 100,
                 exit_percent = 1,
                 signal_only = True):
        """
        Parameters:
            tp_type: string
                can be: rr, atr, dollar, percent
            tp_mult: float
                multiplier for the take profit
            tp_percent: float
                percentage for the take profit
            tp_dollar: float
                dollar amount for the take profit
            exit_percent: float
                amount to subtract from position (represents percentage to be sold when take profit is executed).
                useful for partial take profits.
            signal_only: boolean
                if True, the take profit will only be a signal and not a position change
        """
        self.df = df
        self.tp_type = tp_type
        self.tp_mult = tp_mult
        self.tp_ind_length = tp_ind_length
        self.tp_percent = tp_percent
        self.tp_dollar = tp_dollar
        self.exit_percent = exit_percent
        self.signal_only = signal_only


    def define_tp_pos(self, group, coin):
        """
        This will apply the take profit if a high goes above it to the position column in the dataframe.
        """
        current_pos = group['position', coin].iloc[-1]
        if (group['high', coin] >= group['session_take_profit', coin]).any():
            start = group[group['high', coin] >= group['session_take_profit', coin]].index[0]
            stop = group.index[-2]
            group.loc[start:stop, ("position", coin)] = current_pos - current_pos * self.exit_percent
            return group
        else:
            return group
    
    def define_tp_signal(self, group, coin):
        """
        We are looking for a high that is above the current take profit (of this session)
        """
        group[("exit_signal_tp", coin)] = 0.0  # Initialize with 0
        if (group['high', coin] >= group['session_take_profit', coin]).any():
            start = group[group['high', coin] >= group['session_take_profit', coin]].index[0]
            group.loc[start, ("exit_signal_tp", coin)] = self.exit_percent
            group.loc[start, ('price', coin)] = group.loc[start, ('session_take_profit', coin)]
        return group
        

    def plot_tp(self, df):
        _df = df.copy().unstack()
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
            ax = _df[[['high', coin], ['session_take_profit', coin]]].plot(
                ax=axes[i],
                title=f'{coin} Close and Take Profit',
                color=['blue', 'green'],
                legend = None
            )
            axes[i].set_xlabel('Date')
            axes[i].set_ylabel('Closing Price')
            # axes[i].legend(title='Coin')
            ax2 = ax.twinx()
            _df['position', coin].plot(ax = ax2, alpha = 0.5)

        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def calculate_fixed_tp(self):
        _df = self.df.copy().unstack()

        if self.tp_type.lower() == 'rr':
            if not any('stop_loss' in col for col in _df.columns.get_level_values(0)):
                raise ValueError('No stop loss column found in the dataframe, add a stop loss to add this type of take profit')
            
            for coin in _df.columns.levels[1]:
                _df['take_profit', coin] = _df['close', coin] + self.tp_mult * (_df['close', coin] - _df['stop_loss', coin])
            
            _df = _df.stack(future_stack = True)

        elif self.tp_type.lower() == 'atr':
            if not any('atr' in col for col in _df.columns.get_level_values(0)):
                for coin in _df.columns.get_level_values(1):
                    #Calculate the atr indicator
                    high, low, close = _df['high', coin], _df['low', coin], _df['close', coin]
                    _df['atr', coin] = ta.atr(high, low, close, length=self.tp_ind_length)
                
                #Remove Warm up
                _df = _df.iloc[self.tp_ind_length:]

            _df = _df.stack(future_stack = True)
            _df['take_profit'] = _df['close'] + self.tp_mult * _df['atr']
            
        elif self.tp_type.lower() == 'dollar':
            _df = _df.stack(future_stack = True)
            _df['take_profit'] = _df['close'] + self.tp_dollar
            
        elif self.tp_type.lower() == 'percent':
            _df = _df.stack(future_stack = True)
            _df['take_profit'] = _df['close'] * (1 + self.tp_percent)

        #Apply the session take profit
        _df = _df.unstack()
        for coin in _df.columns.get_level_values(1):
            _df['session_take_profit', coin] = _df['take_profit', coin].groupby(_df['session', coin]).transform('first')

            #Define the take profit position
            if self.signal_only:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_tp_signal(group, coin))
            else:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_tp_pos(group, coin))

        return _df.stack(future_stack = True)

    def calculate_dynamic_tp(self):
        _df = self.df.copy().unstack()

        if self.tp_type.lower() == 'atr':
            if not any('atr' in col for col in _df.columns.get_level_values(0)):
                for coin in _df.columns.get_level_values(1):
                    #Calculate the atr indicator
                    high, low, close = _df['high', coin], _df['low', coin], _df['close', coin]
                    _df['atr', coin] = ta.atr(high, low, close, length=self.tp_ind_length)
                
                #Remove Warm up
                _df = _df.iloc[self.tp_ind_length:]
            
            _df = _df.stack(future_stack = True)
            _df['take_profit'] = _df['close'] + self.tp_mult * _df['atr']

        elif self.tp_type.lower() == 'percent':
            _df = _df.stack(future_stack = True)
            _df['take_profit'] = _df['close'] * (1 + self.tp_percent)

        elif self.tp_type.lower() == 'dollar':
            _df = _df.stack(future_stack = True)
            _df['take_profit'] = _df['close'] + self.tp_dollar

        #Apply the session take profit
        _df = _df.unstack().copy()
        for coin in _df.columns.get_level_values(1):
            _df['session_take_profit', coin] = _df['take_profit', coin].groupby(_df['session', coin]).cummin()

            #Define the take profit position
            if self.signal_only:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_tp_signal(group, coin))
            else:
                _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.define_tp_pos(group, coin))
        
        return _df.stack(future_stack = True)

    def apply_take_profit(self, fixed = True, plot = True):
        """ 
        This function applies the take profit to the ame and calculates the trades, strategy returns, strategy cumulative returns, and sessions.
        it acts as a wrapper for the calculate_fixed_sl function, and make the necessary adjustments after that position column has changed

        Parameters:
            fixed: Boolean
                specifies if the take profit will be fixed (True) or dynamic (False).
                This is essential if we are using tp_type of percent or dollar.
            plot: Boolean
                will plot the closing prices, position, as well as each of the session take profits for each coin
        Note: This assumes calculate fixed tp is called 
        """
        
        _df = self.calculate_fixed_tp() if fixed else self.calculate_dynamic_tp()
        if plot:
            self.plot_tp(_df)
        _df = Calculations().trades(_df)
        _df = Calculations().strategy_returns(_df)
        _df = Calculations().strategy_creturns(_df)
        _df = Calculations().sessions(_df)
        return _df
    


    
            
        

    


    
    