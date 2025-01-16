import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
import quantstats_lumi as qs
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor




class WFO():
    def __init__(self, data, 
                 trading_strategy, 
                 param_grid = {}, 
                 train_size = 1000, 
                 test_size = 1000, 
                 step_size = 1000, 
                 optimize_fn="grid",
                 objective = 'sharpe',
                 opt_freq = 'custom',
                 live = False):
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
        self.data = data
        self.live = live
        self.trading_strategy = trading_strategy
        self.param_grid = param_grid
        if opt_freq == 'custom':
            self.train_size = train_size
        else:
            self.train_size = train_size = self.opt_freq(opt_freq)
        self.test_size = test_size
        self.step_size = step_size
        self.optimize_fn = optimize_fn
        self.objective = objective
        
        if train_size < 0:
            raise ValueError("Train size must be greater than 0.")

        max_param = max(
        param.high if isinstance(param, (Integer, Real)) else max(param) #To handle all different cases
        for param in param_grid.values()
        )

        if step_size + train_size + test_size > len(data):
            raise ValueError("Invalid train, test, or step size. Length of Data: ", len(data), step_size, train_size, test_size)
        elif (train_size < max_param or test_size < max_param):
            raise ValueError("Parameter range exceeds train size or Test size.")
        elif train_size < 1 or test_size < 1 or step_size < 1:
            raise ValueError("Train, test, and step size must be greater than 0.")
        elif step_size < test_size:
            raise ValueError("Step size must be greater than test size.")
        
        if optimize_fn not in ["grid", "gp"]:
            raise ValueError("Invalid optimization function")
        
                             

    #### Helper Methods ####
    def min_train_test_size(self, train_size):
        """
        Calculate the minimum test size based on the train size.
        """

        current_size = train_size
        result = []

        while len(result) == 0 or result is None:
            result = self.trading_strategy(self.data.iloc[:current_size])
            if len(result) != 0:
                break
            current_size *= 2
            

        return current_size - len(result)


        
    def dict_to_param_space(self, param_dict):
        """
        Converts a dictionary of parameters to a list of skopt parameters.
        """
        param_space = []
        for param_name, param_range in param_dict.items():
            if isinstance(param_range, Integer):
                param_space.append(Integer(param_range.low, param_range.high, name=param_name))
            elif isinstance(param_range, Real):
                param_space.append(Real(param_range.low, param_range.high, name=param_name))
            elif isinstance(param_range, Categorical):
                param_space.append(Categorical(param_range.categories, name=param_name))
            elif isinstance(param_range, range):
                param_space.append(Integer(min(param_range), max(param_range), name=param_name))
            else:
                raise ValueError(f"Invalid range for parameter '{param_name}': {param_range}")
        return param_space
        

    def convert_param_space(self, param_space, n_samples=10):
        """
        Converts a parameter dictionary with Integer and Real objects to an iterable format.

        Parameters:
            param_space (dict): Dictionary with Integer or Real objects as values.
            n_samples (int): Number of discrete samples to generate for Real values.
        
        Returns:
            dict: Parameter dictionary with iterable values.
        """
        converted = {}
        for key, value in param_space.items():
            if isinstance(value, Integer):
                # Generate a range of discrete integers
                converted[key] = list(range(value.low, value.high + 1))
            elif isinstance(value, Real):
                # Generate n_samples equally spaced values in the range
                converted[key] = list(np.linspace(value.low, value.high, n_samples))
            else:
                # Assume the parameter is already iterable
                converted[key] = value

        return converted

    

    def split_data(self, data, train_size, test_size, step_size):
        data = data.unstack()
        start = 0
        while start + train_size + test_size <= len(data):
            train = data.iloc[start:start + train_size].stack(future_stack = True)
            test = data.iloc[start + train_size:start + train_size + test_size].stack(future_stack = True)
            print(f"Train: {train.index[0][0]} - {train.index[-1][0]}")
            print(f"Test: {test.index[0][0]} - {test.index[-1][0]}")
            yield train, test
            start += step_size
            
    # def split_data(self, data, train_size, test_size, step_size):
    #     start = 0
    #     while start + train_size + test_size <= len(data):
    #         train = data.iloc[start:start + train_size]
    #         test = data.iloc[start + train_size:start + train_size + test_size]
    #         print(f"Train: {train.index[0][0]} - {train.index[-1][0]}")
    #         print(f"Test: {test.index[0][0]} - {test.index[-1][0]}")
    #         yield train, test
    #         start += step_size


    def objective_function(self, result):
        """
        Calculate the objective function for the optimization.

        Note that we have only included objective functions that we want to maximize.
        """
        if 'strategy' not in result.columns:
            print("No strategy column in result")
            return 0

        strategy = result['strategy'].apply(np.exp) - 1

        if strategy.sum() == 0:
            return 0

        try:
            if self.objective == "multiple":
                creturns = result['strategy'].cumsum().apply(np.exp)
                performance = creturns.iloc[-1]
            elif self.objective == "sharpe":
                performance = qs.stats.sharpe(strategy)
            elif self.objective == "sortino":
                performance = qs.stats.sortino(strategy)
            elif self.objective == "calmar": 
                performance = qs.stats.calmar(strategy)
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
    def optimize_parameters_grid(self, train_data, param_space):
        #Check if the parameter space is iterable (for ParameterGrid compatibility)
        if not any([isinstance(param_range, Iterable) for param_range in param_space.values()]):
            param_grid = self.convert_param_space(param_space, n_samples=20)
        else:
            param_grid = param_space
        
        print([param_space])
        best_params = None
        best_objective = -np.inf
        for params in ParameterGrid(param_grid):
            print(params)
            # print(train_data)
            result = self.trading_strategy(train_data.copy(), params = params)
            if result is None or len(result) == 0:
                continue
            # print(result)
            objective = self.objective_function(result)
            print(objective)
            if objective > best_objective:
                best_objective = objective
                best_params = params
        return best_params

    def optimize_parameters_gp(self, train_data, param_space):

        if isinstance(param_space, dict):
            param_space = self.dict_to_param_space(param_space)
            
        if isinstance(train_data, int):
            train_data = self.data.iloc[-train_data * 2:]

        def objective(param_space):
            result = self.trading_strategy(train_data.copy(), params = param_space)
            # Use negative performance because gp_minimize minimizes
            objective = self.objective_function(result) 
            print(f'In-sample performance: {objective}')
            return -objective if not pd.isnull(objective) else 1e10  # Handle invalid values

        # Run gp_minimize
        result = gp_minimize(
            func=objective,
            dimensions=param_space,
            n_calls=10,  # Number of evaluations
            random_state=42,
        )
        
        if self.live:
            x_values = []
            for _ in range(5):
                result = gp_minimize(objective, param_space, n_calls=10)
                x_values.append(result.x)

            average_x = np.mean(x_values, axis=0) # Average x values
            return {dim.name: val for dim, val in zip(param_space, average_x)}
        
        # Extract the best parameters
        print(f'Best Params: {result.x}')
        best_params = {dim.name: val for dim, val in zip(param_space, result.x)}
        print(f'Best Params Dictionary Created: {best_params}')
        return best_params

   
    def test_strategy(self, test_data, best_params):
        if isinstance(test_data, int):
            test_data = self.data.iloc[-test_data:]
        result = self.trading_strategy(test_data.copy(), **best_params)
        peformance = self.objective_function(result)
        return peformance, result


    # def walk_forward_optimization(self):
    #     """
    #     Perform a walk-forward optimization on a dataset.
    #     """
    #     all_performance = []
    #     all_results = []
    #     for train, test in self.split_data(self.data, self.train_size, self.test_size, self.step_size):
    #         # Optimize on training data
    #         if self.optimize_fn == "grid":
    #             best_params = self.optimize_parameters_grid(train, self.param_grid)
    #         elif self.optimize_fn == "gp":
    #             best_params = self.optimize_parameters_gp(train, self.param_grid)
            
    #         # Test on out-of-sample data
    #         performance, result = self.test_strategy(test, best_params)
    #         print(f"Out-of-sample performance: {performance}")
            
    #         all_performance.append(performance)
    #         all_results.append(result)
        
    #     all_results = pd.concat(all_results)
    #     return all_performance, all_results
    
    def walk_forward_optimization(self):
        """
        Perform a walk-forward optimization on a dataset using parallel threads.
        """
        all_performance = []
        all_results = []

        with ThreadPoolExecutor() as executor:
            futures = []
            for train, test in self.split_data(self.data, self.train_size, self.test_size, self.step_size):
                futures.append(executor.submit(self.perform_walk_forward_iteration, train, test))

            for future in futures:
                performance, result = future.result()
                all_performance.append(performance)
                all_results.append(result)
            
        all_results = pd.concat(all_results)
        return all_performance, all_results

    def perform_walk_forward_iteration(self, train, test):
        if self.optimize_fn == "grid":
            best_params = self.optimize_parameters_grid(train, self.param_grid)
        elif self.optimize_fn == "gp":
            best_params = self.optimize_parameters_gp(train, self.param_grid)

        performance, result = self.test_strategy(test, best_params)
        print(f"Out-of-sample performance: {performance}")
        return performance, result
