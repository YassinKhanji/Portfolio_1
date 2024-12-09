import numpy as np
import datetime as dt
import pandas as pd
import requests

class Calculations():
    def __init__(self):
        pass

    def upsample(self, df, freq='1H'):
        pass

    def trades(self, df):
        """
        Assumes a stacked dataframe
        will generate a column that tracks only when position changes
        """
        df = df.unstack()

        for coin in df['close'].columns:
            df['trades', coin] = df['position', coin].diff().fillna(0).abs()

        #A case where we start with a position, we need to add a trade
        for coin in df['close'].columns:
            if df['position', coin].iloc[0] ==1:
                first_date = df.index[0]
                df.loc[first_date, ('close', coin)]= 1

        df = df.stack(future_stack=True)
        
        return df
    
    def nbr_trades(self, df):
        """
        Assumes a stacked dataframe.
        """
        return df['trades'].sum()
    
    def strategy_returns(self, df):
        """
        Assumes a stacked dataframe

        return a dataframe with column that refers to the strategy returns
        """
        df['strategy'] = df['position'] * df['returns']
        return df
    
    def strategy_creturns(self, df):
        """
        Assumes a stacked dataframe

        return a dataframe with column that refers to the strategy cumulative returns
        """
        df = df.copy().unstack()
        for coin in df['close'].columns:
            df['cstrategy', coin] = df['strategy', coin].cumsum().apply(np.exp)

        df = df.stack(future_stack=True)
        return df


    def sessions(self, df):
        """
        Assumes a stacked dataframe.

        return a dataframe with column that refers to the session, where each session is a unique trade 
            (useful for risk management where we can group by each session and apply a function for each trade individually) 

        Adds/Labels Trading Sessions and their compound returns.
        """
        _df = df.copy().unstack()
        for coin in _df['close'].columns:
            _df['session', coin] = np.sign(_df['trades', coin]).cumsum().shift().fillna(0)
            _df[('session_compound', coin)] = _df['log_return', coin].groupby(_df['session', coin]).cumsum().apply(np.exp) - 1
        
        _df = _df.stack(future_stack=True)

        return _df

