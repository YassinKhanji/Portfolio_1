import numpy as np
import datetime as dt
import pandas as pd
import quantstats_lumi as qs
import math

class Calculations():
    def __init__(self):
        pass

    def downsample(self, df, low_freq='1D'):
        """
        This will be used to resample higher frequency data to lower frequency (e.g. hourly to daily data)
        when performing universe selection (much faster instead of redownloading the daily data)

        Parameters:
            df: Stacked DataFrame
            low_freq: The frequency that the dataframe will be converted to
        """
        pass


    def trades(self, df):
        """
        Assumes a stacked dataframe
        will generate a column that tracks only when position changes
        """
        _df = df.copy().unstack()
        

        for coin in _df['close'].columns:
            full_trade = _df['position', coin].diff().fillna(0).abs() 
            _df['trades', coin] = np.where(full_trade == 1, 1, 0) #This will account when we take partials

        #A case where we start with a position, we need to add a trade
        for coin in _df['close'].columns:
            if _df['position', coin].iloc[0] ==1:
                first_date = _df.index[0]
                _df.loc[first_date, ('trade', coin)]= 1

        _df = _df.stack(future_stack=True)
        
        return _df
    
    def nbr_trades(self, df):
        """
        Takes both a stacked and unstacked dataframe
        """
        _df = df.copy() 

        _df['trades'].sum()

        return _df['trades'].sum()
    
    def calculate_price_returns(self, df):
        """
        Should be called when price column is modified
        This will calculate the log price returns for each coin
        """
        _df = df.copy().unstack()

        for coin in _df.columns.get_level_values(1):
            price = _df['price', coin]
            _df['log_price_return', coin] = np.log(price/price.shift(1))

        
        _df = _df.stack(future_stack=True)

        return _df

    
    def strategy_returns(self, df, costs_per_trade = 0.0):
        """
        Takes both a stacked and unstacked dataframe

        return a dataframe with column that refers to the strategy returns
        """
        _df = df.copy()
        _df = self.calculate_price_returns(_df)
        _df['strategy'] = _df['position'] * _df['log_price_return']
        # df['strategy'] = df['strategy'] - df['trades'] * costs_per_trade #This applies only when there is a dollar cost

        return _df
    
    def strategy_creturns(self, df):
        """
        Takes both a stacked and unstacked dataframe

        return a dataframe with column that refers to the strategy cumulative returns
        """
        _df = df.copy().unstack()

        for coin in _df['close'].columns:
            _df['cstrategy', coin] = _df['strategy', coin].cumsum().apply(np.exp)

        
        _df = _df.stack(future_stack=True)

        return _df


    def sessions(self, df):
        """
        Takes both a stacked and unstacked dataframe
        
        return a dataframe in the same format that it was given
        """
        _df = df.copy().unstack()

        for coin in _df['close'].columns:
            _df['session', coin] = np.sign(_df['trades', coin]).cumsum().shift().fillna(0)
            _df[('session_compound', coin)] = _df['strategy', coin].groupby(_df['session', coin]).cumsum().apply(np.exp)
            _df[('overall_session_return', coin)] = _df['session_compound', coin].groupby(_df['session', coin]).transform(lambda x: x.iloc[-1] - 1)
        
        
        _df = _df.stack(future_stack=True)

        return _df
    
    def merge_cols(self, df, common = 'exit_signal', use_clip = True):
        """
        Assume a stacked dataframe

        parameters:
            df: Stacked DataFrame
            common: The common string that will be used to merge the columns
            use_clip: A boolean that will clip the value between 0 and 1
        """
        cols = [col for col in df.columns if common in col]
        df[common] = df[cols].sum(axis = 1)
        if use_clip:
            df[common] = df[common].clip(0, 1)    
        return df
    
    def update_all(self, df):
        """
        Takes both stacked and unstacked dataframes
        This is a wrapper function that will update all the columns
        """
        stacked = True
        _df = df.copy()
        if isinstance(_df.columns, pd.MultiIndex):
            stacked = True
            _df = _df.stack(future_stack=True)

        _df = self.trades(_df)
        _df = self.strategy_returns(_df)
        _df = self.strategy_creturns(_df)
        _df = self.sessions(_df)

        if not stacked:
            _df = _df.unstack()
        
        return _df

    
class Metrics():

    def __init__(self, df):

        self.df = df.copy().unstack()

        # This would give the number of time periods in a year (e.g. monthly data would be 12)
        self.tp_year = (self.df['close'].count() / ((self.df.index[-1] - self.df.index[0]).days / 365.25))

    def print_performance(self, coin):
        ''' 
        Calculates and prints various Performance Metrics.

        The coin desired must be specified.
        '''
        
        strategy_multiple =     round(self.calculate_multiple(bh = False)[coin], 6)
        bh_multiple =           round(self.calculate_multiple(bh = True)[coin], 6)
        outperf =               round(strategy_multiple - bh_multiple, 6)
        cagr =                  round(self.calculate_cagr()[coin], 6)
        ann_mean =              round(self.calculate_annualized_mean()[coin], 6)
        ann_std =               round(self.calculate_annualized_std()[coin], 6)
        sharpe =                round(self.calculate_sharpe()[coin], 6)
        sortino =               round(self.calculate_sortino()[coin], 6)
        max_drawdown =          round(self.calculate_max_drawdown()[coin], 6)
        calmar =                round(self.calculate_calmar()[coin], 6)
        max_dd_duration =       round(self.calculate_max_dd_duration()[coin], 6)
        avg_dd_duration =       round(self.calculate_avg_dd_duration()[coin], 6)
        kelly_criterion =       round(self.calculate_kelly_criterion()[coin], 6)
        volatility =            round(self.calculate_volatility()[coin], 6)
        win_rate =              round(self.calculate_trades_win_rate()[coin], 2)
        daily_win_rate =        round(self.calculate_daily_win_rate()[coin], 6)
        avg_win =               round(self.calculate_avg_win()[coin], 6)
        avg_loss =              round(self.calculate_avg_loss()[coin], 6)
        avg_return_per_trade =  round(self.calculate_avg_return_per_trade()[coin], 6)
        percentage_expectancy = round(self.calculate_percentage_expectancy()[coin], 6)
        monthly_expectancy =    round(self.calculate_monthly_expectancy()[coin], 6)

        
        print(100 * "=")
        print("\n")
        print(f"All coins: {list(self.df.columns.get_level_values(1).unique())}")
        print("\n")
        print("From all available coins: {} was chosen".format(coin))
        print(100 * "-")
        print("\n")
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
        print("Average Drawdown Duration:   {} Days".format(avg_dd_duration))
        print("Volatility:                  {} Days".format(volatility))
        print("Win Rate:                    {}".format(win_rate))
        print("Daily Win Rate:              {}".format(daily_win_rate))
        print("Average Win:                 {}".format(avg_win))
        print("Average Loss:                {}".format(avg_loss))
        print("Average Return per Trade:    {}".format(avg_return_per_trade))
        print("Percentage Expectancy:       {}".format(percentage_expectancy))
        print("Monthly Expectancy:          {}".format(monthly_expectancy))
        print("Kelly Criterion:             {}".format(kelly_criterion))
        
        print(100 * "=")

    def calculate_multiple(self, bh = False):
        """
        The multiple is a measure of the total return of the strategy.

        Typically >1 is considered good, <1 should not be considered.
        """


        results = {}
        for coin in self.df['strategy'].columns:
            if bh:
                series = self.df['returns', coin]
            else:
                series = self.df['strategy', coin]
                
            results[coin] = np.exp(series.sum()) 

        return results

    def calculate_annualized_mean(self):
        """
        The annualized mean is a measure of the average return of the strategy.

        Typically >10% is considered good for Bull Markets, >2 - 3% for Bear Markets.
        """ 
    
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = self.df['strategy', coin].mean() * self.tp_year[coin]

        return results
    
    def calculate_annualized_std(self):
        """
        The standard deviation is a measure of the dispersion of returns.

        Typically <20% is considered good for Bull Markets, <10% for Bear Markets (even for aggressive portfolios)
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = self.df['strategy', coin].std() * np.sqrt(self.tp_year[coin])

        return results

    def calculate_max_dd_duration(self):
        """
        The maximum drawdown duration is the time it takes for the strategy to recover from a drawdown (does not have to be the maximum)

        Typically 3 - 6 months is considered to be good for a maximum duration
        """
        results = {}
        for coin in self.df['strategy'].columns:
            series = self.df['strategy', coin]
            creturns = series.cumsum().apply(np.exp)
            cummax = creturns.cummax()
            drawdown = (cummax - creturns)/cummax

            begin = drawdown[drawdown == 0].index
            end = begin[1:]
            end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
            periods = end - begin
            results[coin] = periods.max().days

        return results

    def calculate_avg_dd_duration(self):
        """
        The average drawdown duration is the average time it takes for the strategy to recover from a drawdown.

        Typically 1 - 3 months is considered to be good for an average duration
        
        """
        results = {}
        for coin in self.df['strategy'].columns:
            series = self.df['strategy', coin]
            creturns = series.cumsum().apply(np.exp)
            cummax = creturns.cummax()
            drawdown = (cummax - creturns)/cummax

            begin = drawdown[drawdown == 0].index
            end = begin[1:]
            end = end.append(pd.DatetimeIndex([drawdown.index[-1]]))
            periods = end - begin
            results[coin] = periods.mean().days

        return results
            

    def calculate_cagr(self):
        """"
        Compound Annual Growth Rate (CAGR) measures the mean annual growth rate of an investment
         over a specified time period longer than one year.

        CAGR is really dependent on the investment vehicule. For example, for a stock, 10%-15% is considered good.
        For crypto, it is much higher, typically >50%.
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.cagr(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_volatility(self):
        """
        Volatility is a measure of the dispersion of returns. It is a measure of the risk of the strategy.

        Investors typically prefer lower volatility between 10-15% for the overall portfolio
        """
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = qs.stats.volatility(self.df['strategy', coin].apply(np.exp) - 1)

        return results
    
    def calculate_sharpe(self):
        """
        Sharpe Ratio measures the excess return (or risk premium) per unit of total volatility (both upside and downside)
        Some may refer to it as the smoothness of the equity curve.

        Typically >1 is considered good for both trend-following and mean-reversion. <0.5 should not be considered
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
        results = {}
        for coin in self.df['strategy'].columns:
            active_trades = self.df[self.df['position', coin] == 1]
            results[coin] = active_trades['overall_session_return', coin].mean()

        return results

    def calculate_percentage_expectancy(self):
        """
        This calculates the expectancy of the strategy per coin.
        Expectancy = (Win Rate * Average Win) - (Losing Rate * Average Loss)

        Typically any expectancy greater than 0 is considered good (it depends on the number of trades).
        """
        results = {}
        for coin in self.df['strategy'].columns:
            win_rate = self.calculate_trades_win_rate()[coin]
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
        for coin in self.df['strategy'].columns:
            total_trades = math.ceil(self.df['session', coin].iloc[-1] / 2)
            results[coin] = self.calculate_percentage_expectancy()[coin] * (total_trades / 12)

        return results