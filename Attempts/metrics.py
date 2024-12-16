class Metrics():

    """
    Note: Although we could have used the quantstats library, we decided to implement our own metrics:
        1. For the sake of learning
        2. Because some metrics are not well implemented in quantstats. For example, win_rate is calculated using daily returns
            which is not the best way to calculate it. We used the number of trades instead.
    """

    def __init__(self, df):
        self.df = df.copy().unstack()

        # This would give the number of time periods in a year (e.g. monthly data would be 12)
        self.tp_year = (self.df['close'].count() / ((self.df.index[-1] - self.df.index[0]).days / 365.25))

    def calculate_multiple(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = np.exp(self.df['strategy', coin].sum())
        return results

    def calculate_cagr(self):
        results = {}
        for coin in self.df['strategy'].columns:
            series = self.df['strategy', coin]
            results[coin] = np.exp(series.sum())**(1/((series.index[-1] - series.index[0]).days / 365.25)) - 1
        return results

    def calculate_annualized_mean(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = self.df['strategy', coin].mean() * self.tp_year
        return results

    def calculate_annualized_std(self):
        results = {}
        for coin in self.df['strategy'].columns:
            results[coin] = self.df['strategy', coin].std() * np.sqrt(self.tp_year)
        return results

    def calculate_sharpe(self):
        results = {}
        for coin in self.df['strategy'].columns:
            series = self.df['strategy', coin]
            if series.std() == 0:
                results[coin] = np.nan
            else:
                results[coin] = series.mean() / series.std() * np.sqrt(self.tp_year)
        return results

    def calculate_sortino(self):
        results = {}
        for coin in self.df['strategy'].columns:
            series = self.df['strategy', coin]
            excess_returns = (series - 0)
            downside_deviation = np.sqrt(np.mean(np.where(excess_returns < 0, excess_returns, 0)**2))
            if downside_deviation == 0:
                results[coin] = np.nan
            else:
                results[coin] = (series.mean() - 0) / downside_deviation * np.sqrt(self.tp_year)
        return results

    def calculate_max_drawdown(self):
        results = {}
        for coin in self.df['strategy'].columns:
            series = self.df['strategy', coin]
            creturns = series.cumsum().apply(np.exp)
            cummax = creturns.cummax()
            drawdown = (cummax - creturns)/cummax
            results[coin] = drawdown.max()
        return results

    def calculate_calmar(self):
        results = {}
        for coin in self.df['strategy'].columns:
            max_dd = self.calculate_max_drawdown()[coin]
            if max_dd == 0:
                results[coin] = np.nan
            else:
                cagr = self.calculate_cagr()[coin]
                results[coin] = cagr / max_dd
        return results

    def calculate_max_dd_duration(self):
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

    def calculate_kelly_criterion(self):
        results = {}
        for coin in self.df['strategy'].columns:
            series = np.exp(self.df['strategy', coin]) - 1
            if series.var() == 0:
                results[coin] = np.nan
            else:
                results[coin] = series.mean() / series.var()
        return results
    
    def calculate_avg_dd_duration(self):
        """
        Assumes an unstacked dataframe
        """
        for coin in self.df['close'].columns:
            self.df['avg_dd_duration', coin] = (self.df['cstrategy', coin] / self.df['cstrategy', coin].cummax() - 1).abs().mean()

        return self.df.stack(future_stack=True)

    def calculate_avg_win(self):
        pass

    def calculate_avg_loss(self):
        pass

    def calculate_win_loss_ratio(self):
        pass

    def calculate_avg_trade_return(self):
        pass

    def calculate_win_rate(self):
        pass

    def print_performance(self, leverage = False):
        ''' Calculates and prints various Performance Metrics.
        '''
        
        data = self.df.copy()
        
        to_analyze = data['strategy']
        
        strategy_multiple = round(self.calculate_multiple(to_analyze), 6)
        bh_multiple =       round(self.calculate_multiple(data.returns), 6)
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
