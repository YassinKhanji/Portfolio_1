import numpy as np
import datetime as dt
import pandas as pd
import requests
import quantstats_lumi as qs

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
                df.loc[first_date, ('trade', coin)]= 1

        df = df.stack(future_stack=True)
        
        return df
    
    def nbr_trades(self, df):
        """
        Assumes a stacked dataframe.
        """
        return df['trades'].sum()
    
    def strategy_returns(self, df, costs_per_trade = 0.0):
        """
        Assumes a stacked dataframe

        return a dataframe with column that refers to the strategy returns
        """
        df['strategy'] = df['position'] * df['returns']
        # df['strategy'] = df['strategy'] - df['trades'] * costs_per_trade #This applies only when there is a dollar cost
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
        _df = df.copy().unstack()
        for coin in _df['close'].columns:
            _df['session', coin] = np.sign(_df['trades', coin]).cumsum().shift().fillna(0)
            _df[('session_compound', coin)] = _df['strategy', coin].groupby(_df['session', coin]).cumsum().apply(np.exp)
            _df[('overall_session_return', coin)] = _df['session_compound', coin].groupby(_df['session', coin]).transform(lambda x: x.iloc[-1] - 1)
        
        _df = _df.stack(future_stack=True)

        return _df

class Metrics():

    def __init__(self, df):

        self.df = df

    def print_performance(self, leverage = False):
        ''' Calculates and prints various Performance Metrics.
        '''
        
        data = self.df.copy()
        
        to_analyze = data['strategy']
        
        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple =       round(self.calculate_multiple(data['returns']), 6)
        outperf =           round(strategy_multiple - bh_multiple, 6)
        cagr =              round(self.calculate_cagr(to_analyze), 6)
        ann_mean =          round(self.calculate_annualized_mean(to_analyze), 6)
        ann_std =           round(self.calculate_annualized_std(to_analyze), 6)
        sharpe =            round(self.calculate_sharpe(to_analyze), 6)
        sortino =           round(self.calculate_sortino(to_analyze), 6)
        max_drawdown =      round(self.calculate_max_drawdown(to_analyze), 6)
        calmar =            round(self.calculate_calmar(to_analyze), 6)
        max_dd_duration =   round(self.calculate_max_dd_duration(to_analyze), 6)
        kelly_criterion =   round(self.calculate_kelly_criterion(to_analyze), 6)
        
        print(100 * "=")
        print("SIMPLE CONTRARIAN STRATEGY | INSTRUMENT = {} | Freq: {} | WINDOW = {}".format(self.symbol, self.freq, self.window))
        print(100 * "-")
        #print("\n")
        print("PERFORMANCE MEASURES:")
        print("\n")
        print("Multiple (Strategy):         {}".format(strategy_multiple))
        print("Multiple (Buy-and-Hold):     {}".format(bh_multiple))
        print(38 * "-")
        print("Out-/Underperformance:       {}".format(outperf))
        print("\n")
        print("CAGR:                        {}".format(cagr))
        print("Annualized Mean:             {}".format(ann_mean))
        print("Annualized Std:              {}".format(ann_std))
        print("Sharpe Ratio:                {}".format(sharpe))
        print("Sortino Ratio:               {}".format(sortino))
        print("Maximum Drawdown:            {}".format(max_drawdown))
        print("Calmar Ratio:                {}".format(calmar))
        print("Max Drawdown Duration:       {} Days".format(max_dd_duration))
        print("Kelly Criterion:             {}".format(kelly_criterion))
        
        print(100 * "=")

    def calculate_multiple(self):
        pass

    def calculate_annualized_mean(self):
        pass
    
    def calculate_annualized_std(self):
        pass 

    def calculate_max_dd_duration(self):
        pass

    def calculate_cagr(self):
        """"
        We are going to be using quantstats 
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.cagr(self.df['cstrategy', coin])

        return results
    
    def calculate_volatility(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.volatility(self.df['cstrategy', coin])

        return results
    
    def calculate_sharpe(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.sharpe(self.df['cstrategy', coin])

        return results
    
    def calculate_sortino(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.sortino(self.df['cstrategy', coin])

        return results
    
    def calculate_max_drawdown(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.max_drawdown(self.df['cstrategy', coin])

        return results
    
    def calculate_calmar(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.calmar(self.df['cstrategy', coin])

        return results
    
    def calculate_kelly_criterion(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.kelly_criterion(self.df['cstrategy', coin]) 
        
        return results
    
    def calculate_win_rate(self):
        results = {}
        #Note that the winrate that qs uses is using the daily returns. However, we are looking for returns per trade.
        # Therefore, this should be changed. 
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.win_rate(self.df['strategy', coin])

        return results
    
    def calculate_avg_win(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.avg_win(self.df['strategy', coin])

        return results
    
    def calculate_avg_loss(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.avg_loss(self.df['strategy', coin])

        return results
    
    def calculate_avg_trade_return(self):
        pass

    def calculate_expectancy(self):
        pass