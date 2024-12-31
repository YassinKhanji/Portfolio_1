import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args
import yfinance as yf



class WFO():
    def __init__(self, data, trading_strategy, param_grid, train_size, test_size, step_size, optimize_fn="grid"):
        """
        This class performs a walk-forward optimization on a trading strategy.

        Parameters:
        data (pd.DataFrame): The historical data to be used for backtesting.
        trading_strategy (object): The trading strategy to be optimized.
        param_grid (dict): The grid of parameters to be optimized.
        train_size (int): The number of data points to be used for training.
        test_size (int): The number of data points to be used for testing.
        step_size (int): The number of data points to step forward in each iteration.
        """
        self.data = data
        self.trading_strategy = trading_strategy
        self.param_grid = param_grid
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.optimize_fn = optimize_fn

        max_param = max(
        param.high if isinstance(param, (Integer, Real)) else max(param) #To handle all different cases
        for param in param_grid.values()
        )
        
        if step_size + train_size + test_size > len(data):
            raise ValueError("Invalid train, test, or step size.")
        if (train_size > max_param or test_size > max_param):
            raise ValueError("Parameter range exceeds train size or Test size.")
        if optimize_fn not in ["grid", "gp"]:
            raise ValueError("Invalid optimization function")
        
                             

    #### Helper Methods ####
    def dict_to_param_space(self, param_dict):
        param_space = []
        for param_name, param_range in param_dict.items():
            if isinstance(param_range, tuple) and len(param_range) == 2:
                if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                    # Create an Integer parameter range
                    param_space.append(Integer(param_range[0], param_range[1], name=param_name))

                elif isinstance(param_range[0], float) and isinstance(param_range[1], float):
                    # Create a Real parameter range for floats
                    param_space.append(Real(param_range[0], param_range[1], name=param_name))

                elif isinstance(param_range[0], Categorical) and isinstance(param_range[1], Categorical):
                    # Create a Categorical parameter range
                    param_space.append(Categorical(param_range, name=param_name))

            elif isinstance(param_range, range):
                # Convert range to min and max bounds
                param_space.append(Integer(min(param_range), max(param_range), name=param_name))
            else:
                raise ValueError(f"Invalid range for parameter '{param_name}': {param_range}")
        return param_space
    

    def split_data(self, data, train_size, test_size, step_size):
        start = 0
        while start + train_size + test_size <= len(data):
            train = data.iloc[start:start + train_size]
            test = data.iloc[start + train_size:start + train_size + test_size]
            yield train, test
            start += step_size


    #### Optimization Methods ####
    def optimize_parameters_grid(self, train_data, param_grid):
        best_params = None
        best_performance = -np.inf
        for params in ParameterGrid(param_grid):
            result = self.trading_strategy(train_data.copy(), **params)
            performance = result['creturns'].iloc[-1]  # Get the last value of cumulative returns
            if performance > best_performance:
                best_performance = performance
                best_params = params
        return best_params

    def optimize_parameters_gp(self, train_data, param_space):

        if isinstance(param_space, dict):
            param_space = self.dict_to_param_space(param_space)

        @use_named_args(param_space)
        def objective(**params):
            result = self.trading_strategy(train_data.copy(), **params)
            # Use negative performance because gp_minimize minimizes
            performance = result["creturns"].iloc[-1]
            return -performance if not pd.isnull(performance) else np.inf  # Handle invalid values

        # Run gp_minimize
        result = gp_minimize(
            func=objective,
            dimensions=param_space,
            n_calls=50,  # Number of evaluations
            random_state=42,
        )
        
        # Extract the best parameters
        best_params = {dim.name: val for dim, val in zip(param_space, result.x)}
        return best_params

   
    def test_strategy(self, test_data, best_params):
        result = self.trading_strategy(test_data.copy(), **best_params)
        if "creturns" in result.columns:
            return result["creturns"].iloc[-1]
        else:
            return np.nan


    def walk_forward_optimization(self):
        """
        Perform a walk-forward optimization on a dataset.
        """
        results = []
        for train, test in self.split_data(self.data, self.train_size, self.test_size, self.step_size):
            # Optimize on training data
            if self.optimize_fn == "grid":
                best_params = self.optimize_parameters_grid(train, self.param_grid)
            elif self.optimize_fn == "gp":
                best_params = self.optimize_parameters_gp(train, self.param_grid)
            
            # Test on out-of-sample data
            performance = self.test_strategy(test, best_params)
            print(f"Out-of-sample performance: {performance}")
            
            results.append(performance)
        return results
