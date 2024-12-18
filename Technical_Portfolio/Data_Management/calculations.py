import numpy as np
import datetime as dt
import pandas as pd
import requests
import quantstats_lumi as qs
import math

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
        """
        Expectations for the multiple:
        Trend-Following: Positive and steady annual returns. Aim for 10-15% annualized returns.
        Mean Reversion: Positive returns with lower volatility, 8-12% annualized returns.
        Portfolio: 10% or higher for a well-balanced portfolio.
        Bullish Market: Higher returns expected (15-20%).
        Bearish Market: Lower returns, possibly negative for trend-following.
        """
        pass

    def calculate_annualized_mean(self):
        """
        Expectations for the annualized mean:
        Trend-Following: Moderate risk, aim for 10-15% annualized volatility.
        Mean Reversion: Lower risk, 8-12% annualized volatility.
        Portfolio: Should be around 10-15% for diversification.
        Bullish Market: Typically lower volatility (10-12%).
        Bearish Market: Higher volatility during drawdowns (15-20%).
        """ 
    
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
            results[coin] = qs.stats.cagr(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_volatility(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.volatility(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_sharpe(self):
        """
        
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.sharpe(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_sortino(self):
        """
        Sortino ratio is a variation of the Sharpe ratio that only factors in the downside risk.

        Typically >1.5 for both trend-following and mean-reversion.
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.sortino(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_max_drawdown(self):
        """
        Depends on the strategy used. Typically <20% for trend-following, <10% for mean-reversion.
        Depends on the investor's risk tolerance. Typically <30% should not be considered.
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.max_drawdown(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_calmar(self):
        """
        calmar = Annualized return / Max Drawdown

        Typically >1 for trend-following, >1.5 for mean-reversion.

        A calmar ration less than 1 should not be considered.
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.calmar(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_kelly_criterion(self):
        """
        This is a measure of the optimal bet size that a trader should make.
        The higher, the better.

        Typically Values should be positive, generally between 0.1 and 0.5.
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.kelly_criterion(self.df['strategy', coin].apply(np.exp) - 1) 
        
        return results
    
    def calculate_trades_win_rate(self):
        """
        Typically >50% for mean-reversion and anything for trend-following.
        (depends on the profit factor)
        """
        results = {}
        for coin in self.df['strategy'].columns:
            winning_trades = (self.df['overall_session_return', coin] > 0).sum()
            total_trades = math.ceil(self.df['session', coin].iloc[-1] / 2) #we divide by 2 because a session can indicates being in a trade and not being in a trade
            results[coin] = winning_trades / total_trades

        return results
    
    def calculate_daily_win_rate(self):
        """
        Typically >50% in both strategies.
        """
        results = {}
        #Note that the winrate that qs uses is using the daily returns. However, we are looking for returns per trade.
        # Therefore, this should be changed. 
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.win_rate(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_avg_win(self):
        """
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.avg_win(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_avg_loss(self):
        """
        """
        # this is a comment
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.avg_loss(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_avg_return_per_trade(self):
        """
        This reflects the average return per trade = total PnL / number of trades

        Typically Positive, typically >2% for trend-following, >1% for mean-reversion.
        
        """
        pass

    def calculate_percentage_expectancy(self):
        """
        This calculates the expectancy of the strategy per coin.
        Expectancy = (Win Rate * Average Win) - (Losing Rate * Average Loss)

        Typically any expectancy greater than 0 is considered good (it depends on the number of trades).
        """
        results = {}
        for coin in self.df['strategy'].columns:
            win_rate = self.calculate_trades_win_rate(self.df)[coin]
            losing_rate = 1 - win_rate
            average_win = self.df[(self.df['position', coin] == 1) & (self.df['overall_session_return', coin] >= 0)]['overall_session_return', coin].mean()
            average_loss = self.df[(self.df['position', coin] == 1) & (self.df['overall_session_return', coin] <= 0)]['overall_session_return', coin].mean()
            expectancy = (win_rate * average_win) - (losing_rate * average_loss)
            results[coin] = expectancy

        return results  
    
    def calculate_monthly_expectancy(self):
        """
        Typically: 4-6% monthly in bullish, lower for mean-reversion in bearish.
        """
        results = {}
        total_trades = math.ceil(self.df['session', coin].iloc[-1] / 2)
        for coin in self.df['strategy'].columns:
            results[coin] = self.calculate_percentage_expectancy(self.df)[coin] * (total_trades / 12)

        return results