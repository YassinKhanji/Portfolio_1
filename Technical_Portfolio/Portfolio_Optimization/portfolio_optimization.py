import numpy as np
import pandas as pd
from scipy.optimize import minimize
import quantstats_lumi as qs

class Portfolio_Optimization():
    def __init__(self, strategies, 
                 train_size = 1000, 
                 test_size = 1000, 
                 step_size = 1000, 
                 objective = 'sharpe',
                 opt_freq = 'custom'):
        """
        This class performs a walk-forward optimization on a trading strategy.

        Parameters:
        data (pd.DataFrame): The historical data to be used for backtesting.
        trading_strategy (object): The trading strategy to be optimized.
        param_grid (dict): The grid of parameters to be optimized.
        train_size (int): The number of data points to be used for training.
        test_size (int): The number of data points to be used for testing.
        step_size (int): The number of data points to step forward in each iteration.
        optimize_fn (str): The optimization function to use ("grid" or "gp").
        objective (str): The objective function to maximize ("sharpe", "sortino", "calmar", "multiple").
        opt_period (str): The period to optimize over ['custom', 'daily', 'weekly', 'quarterly', 'semi-annually', 'yearly'].
        """
        self.strategies = strategies
        if opt_freq == 'custom':
            self.train_size = train_size
        else:
            self.train_size = train_size = self.opt_freq(opt_freq)
        self.test_size = test_size
        self.step_size = step_size
        self.objective = objective
        
        
        if step_size + train_size + test_size > len(strategies):
            raise ValueError("Invalid train, test, or step size.")
        elif train_size < 1 or test_size < 1 or step_size < 1:
            raise ValueError("Train, test, and step size must be greater than 0.")
        elif test_size < train_size:
            raise ValueError("Test size must be greater or equal to train size.")
        
                            
    #### Helper Methods ####
    def split_data(self, data, train_size, test_size, step_size):
        start = 0
        while start + train_size + test_size <= len(data):
            train = data.iloc[start:start + train_size]
            test = data.iloc[start + train_size:start + train_size + test_size]
            yield train, test
            start += step_size
            
    def calculate_returns(self, weights, log_rets):
        return np.dot(weights, log_rets.T)
    
    def bounds(self, N):
        # Initialize an empty list to collect the inner tuples
        collected_tuples = []
            
        # Iterate n times to create n inner tuples
        for _ in range(N):
            # Append the specific numbers as a tuple to the list
            collected_tuples.append((0, 1))
            
            # Convert the list of tuples to a tuple of tuples and return it
        return tuple(collected_tuples)
            

    def objective_function(self, weights, train_data):
        """
        Calculate the objective function for the optimization.

        Note that we have only included objective functions that we want to maximize.
        """

        strategy_combined = pd.DataFrame(self.calculate_returns(weights, train_data))[0]

        if strategy_combined.sum() == 0:
            return 0

        try:
            if self.objective == "multiple":
                creturns = strategy_combined.cumsum().apply(np.exp)
                performance = creturns.iloc[-1]
            elif self.objective == "sharpe":
                performance = qs.stats.sharpe(strategy_combined)
            elif self.objective == "sortino":
                performance = qs.stats.sortino(strategy_combined)
            elif self.objective == "calmar": 
                performance = qs.stats.calmar(strategy_combined)
            else:
                raise ValueError("Invalid objective function")
        except Exception as e:
            print(f"Error calculating performance: {e}")
            performance = 0

        return performance
    
    def opt_freq(self, opt_freq):
        time_diff = self.data.unstack().index.get_level_values(0)[1] - self.data.unstack().index.get_level_values(0)[0]

        if opt_freq == 'daily':
            return pd.Timedelta('1 day') // time_diff
        elif opt_freq == 'weekly':
            return pd.Timedelta('1 w') // time_diff
        elif opt_freq == 'monthly':
            return pd.Timedelta('1 m') // time_diff
        elif opt_freq == 'quarterly':
            return pd.Timedelta('3 m') // time_diff
        elif opt_freq == 'semi-annually':
            return pd.Timedelta('6 m') // time_diff
        elif opt_freq == 'yearly':
            return pd.Timedelta('1 y') // time_diff
        else:
            raise ValueError("Invalid optimization frequency")
        


    #### Optimization Methods ####
    
    def optimize_weights_minimize(self, train_data):
        """
        Optimize the weights of a trading strategy using Bayesian optimization.
        """
        equal_weights = np.array([1 / len(self.strategies.columns)] * len(self.strategies.columns))
        bounds = self.bounds(len(self.strategies.columns))
        
        sum_constraint = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
        starting_guess = equal_weights
        
        # Call the minimize function
        result = minimize(
        fun=lambda weights: -self.objective_function(weights, train_data),  # Minimize negative performance
        x0=starting_guess,  # Starting point for optimization
        bounds=bounds,  # Bounds for each weight
        constraints=sum_constraint  # Constraint: weights sum to 1
        )
        
        return result.x

   
    def test_weights(self, weights, test_data):
        
        result = self.calculate_returns(weights, test_data)
        peformance = self.objective_function(weights, test_data)
        return peformance, result


    def walk_forward_optimization(self):
        """
        Perform a walk-forward optimization on a dataset.
        """
        all_performance = []
        all_results = []
        for train, test in self.split_data(self.strategies, self.train_size, self.test_size, self.step_size):
            # Optimize on training data    
            weights = self.optimize_weights_minimize(train)
            
            # Test on out-of-sample data
            performance, result = self.test_weights(test, weights)
            print(f"Out-of-sample performance: {performance}")
            
            all_performance.append(performance)
            all_results.append(pd.DataFrame(result))
        
        all_results = pd.concat(all_results).reset_index(drop = True)
        return all_performance, all_results
