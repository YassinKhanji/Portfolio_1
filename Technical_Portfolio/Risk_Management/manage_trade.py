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
from tail_risk import Stop_Loss, Take_Profit


class Manage_Trade():
    def __init__(self, df):
        """
        df: Stacked dataframe with the session stop loss calculated
        method: The method used to calculate the actual allocation
            Available methods: 
                erw (Equal Risk Weighting, using session stop loss),
                ew (Equal volatility Weighting),
        strategy: The strategy used to calculate the actual allocation
        max_percent_risk: Maximum percent risk per trade
        max_dollar_allocation: Maximum dollar allocation per trade
        """
        self.df = df


    ############# Helper Functions #############
    def def_trade_size(self, group, coin):
        """
        Assumes a stacked dataframe with the session compound, session stop loss, and actual allocation calculated
        """
        allocation_for_trade = group['actual_allocation', coin].iloc[0]

        #Calculate the current allocation
        group['current_allocation', coin] = group['position', coin] *group['session_compound', coin] * allocation_for_trade

        #The reason why we are doing the above, is because position can be any number between 0 and 1 if we have taken partials during the trade
        # depending on the strategy used. So we need to multiply the position by the actual allocation to get the current allocation.
        # We also multiply by the session compound to account for the how well the trade is doing.
        return group
    

    ############# Weighting Methods #############
    def erw_actual_allocation(self, max_percent_risk, max_dollar_allocation):
        """
        Assume a stacked dataframe with the session stop loss calculated.
        Uses the distance to the session stop loss to calculate the actual allocation.
        ERW stands for Equal Risk Weighting
        """
        _df = self.df.copy()
        
        #distance to session stop loss, to apply it for all types of stop losses (normalized)
        _df['distance_to_stop'] = (_df['close'] - _df['session_stop_loss']) / _df['close']

        #Calcualte the max dollar risk per trade
        max_dollar_risk_per_trade = max_percent_risk * max_dollar_allocation


        _df['actual_allocation'] = max_dollar_risk_per_trade / _df['distance_to_stop']
        _df['actual_allocation'] = _df['actual_allocation'].clip(upper = max_dollar_allocation, lower = 0)
        #Since we are not gonna use any leverage, we can't allocate more than the max_dollar_allocation
        #we can't allocate less than 0

        _df = _df.unstack()
        for coin in _df.columns.get_level_values(1):
            _df = _df.groupby(_df['session', coin], group_keys=False).apply(lambda group: self.def_trade_size(group, coin))

        self.df = _df.stack(future_stack = True)

        return _df.stack(future_stack = True)
    

    def risk_parity():
        pass


    ############# Weighting Strategies #############
    def martingale():
        pass

    def anti_martingale():
        pass
    