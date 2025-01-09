import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

class Returns_Metrics():
    def __init__(self, prices):
       self.prices = prices
    
    # 2. Define the Performance Metric Functions
    def max_drawdown_fnct(self):
        """Calculate the maximum drawdown (peak to trough)."""
        drawdowns = (self.prices / self.prices.cummax() - 1)
        return drawdowns.min()

    def average_drawdown(self):
        """Calculate the average drawdown (peak to trough)."""
        drawdowns = (self.prices / self.prices.cummax() - 1)
        return drawdowns.mean()

    def average_drawdown_duration(self):
        """Calculate the average duration of drawdowns."""
        drawdowns = (self.prices / self.prices.cummax() - 1)
        drawdown_durations = []
        drawdown_start = None
        for i in range(1, len(drawdowns)):
            if drawdowns[i] < 0:
                if drawdown_start is None:
                    drawdown_start = i
            else:
                if drawdown_start is not None:
                    drawdown_durations.append(i - drawdown_start)
                    drawdown_start = None
        return np.mean(drawdown_durations) if drawdown_durations else 0

    def sharpe_ratio_fnct(self, risk_free_rate=0):
        """Calculate the Sharpe ratio."""
        returns = self.prices.pct_change().dropna()
        excess_returns = returns - risk_free_rate
        return excess_returns.mean() / excess_returns.std() if excess_returns.std() != 0 else 0
    
    def var(self, confidence_level):
        return np.quantile(self.prices, 1 - confidence_level)
    
    def cvar(self, confidence_level):
        var = np.quantile(self.prices, 1 - confidence_level)
        return np.mean(self.prices[self.prices <= var])
    
    def confidence_interval_fnct(self, metric_series, confidence_level):
        lower_bound = np.percentile(metric_series, (1-confidence_level)/2 * 100)
        upper_bound = np.percentile(metric_series, (1+confidence_level)/2 * 100)
        return lower_bound, upper_bound
    
    
class Stress_Test():
    def __init__(self, returns, num_simulations, confidence_level):
        self.returns = returns
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        
    def normal_sims(self):
        mu, sigma = self.returns.mean(), self.returns.std()
        normal_prices_df = pd.DataFrame()
        for i in range(self.num_simulations):
            sim_rets = np.random.normal(mu, sigma, 252)
            sim_prices = np.exp(sim_rets.cumsum())
            normal_prices_df[i] = sim_prices
            plt.axhline(initial_price, c = 'k')
            plt.plot(sim_prices)
        return normal_prices_df
    
    def t_sims(self):
        df, loc, scale = stats.t.fit(self.returns)
        simulated_returns = stats.t.rvs(df, loc=loc, scale=scale, size=(self.num_simulations, len(self.returns)))
        simulated_cumulative_returns = np.exp(simulated_returns.cumsum(axis=1))
        simulated_prices_df = pd.DataFrame(simulated_cumulative_returns).transpose()
        for i in range(self.num_simulations):
            plt.plot(simulated_cumulative_returns[i])
        return simulated_prices_df
    
    def block_bootstrap(self, block_size):
        resampled_series = []
        for _ in range(self.num_simulations):
            blocks = [self.returns[i:i + block_size] for i in range(len(self.returns) - block_size + 1)]
            sampled_blocks = [blocks[np.random.randint(0, len(blocks))] for _ in range(len(self.returns) // block_size + 1)]
            resampled = np.concatenate(sampled_blocks)[:len(self.returns)]
            resampled_cum = np.exp(resampled.cumsum())
            resampled_series.append(resampled_cum)
            resampled_df = pd.DataFrame(resampled_series).transpose()
            for i in range(self.num_simulations):
                plt.plot(resampled_df[i])
        return resampled_df
    
    def metrics_df_fnct(self, sims_df):
        metrics_dict = {
            'max_drawdown': [],
            'avg_drawdown': [],
            'avg_drawdown_duration': [],
            'sharpe_ratio': [],
            'var': [],
            'cvar': []
        }
        

        # Calculate metrics for each simulation path
        for i in range(self.num_simulations):
            sim_prices = sims_df.iloc[:, i]  # Get the i-th column (price path)
            metrics = Returns_Metrics(sim_prices)
            metrics_dict['max_drawdown'].append(metrics.max_drawdown_fnct())
            metrics_dict['avg_drawdown'].append(metrics.average_drawdown())
            metrics_dict['avg_drawdown_duration'].append(metrics.average_drawdown_duration())
            metrics_dict['sharpe_ratio'].append(metrics.sharpe_ratio_fnct())
            metrics_dict['var'].append(metrics.var(self.confidence_level))
            metrics_dict['cvar'].append(metrics.cvar(self.confidence_level))
        
        # Convert metrics to a DataFrame for easy handling
        metrics_df = pd.DataFrame(metrics_dict)
        return metrics_df
    
    def score_strategy(self, metrics_df):
        mdd = metrics_df['max_drawdown'].mean()
        avg_dd = metrics_df['avg_drawdown'].mean()
        sharpe = metrics_df['sharpe_ratio'].mean()
        recovery = metrics_df['avg_drawdown_duration'].mean()
        var = metrics_df['var'].mean()
        cvar = metrics_df['cvar'].mean()
        
        
        weights = [1 / len(metrics_df.columns)] * len(metrics_df.columns)
        mdd_score = 100 - (abs(mdd) / 0.50) * 100
        avg_dd_score = 100 - (abs(avg_dd) / 0.25) * 100
        avg_dd_duration_score = 100 - (abs(recovery) / 50) * 100
        sharpe_score = (sharpe) / 0.5 * 100
        var_score = 100 - abs(var) / 0.2 * 100
        cvar_score = 100 - abs(cvar) / 0.2 * 100
        
        total_score = np.dot(weights, [mdd_score, avg_dd_score, avg_dd_duration_score, sharpe_score, var_score, cvar_score])
        return total_score    
        