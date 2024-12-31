import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Integer, Categorical, Real
from skopt.utils import use_named_args




#Helper method that converts a dictionary of parameters to a list of tuples
def dict_to_param_space(param_dict):
    param_space = []
    for param_name, param_range in param_dict.items():
        if isinstance(param_range, tuple) and len(param_range) == 2:
            if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                # Create an Integer parameter range
                param_space.append(Integer(param_range[0], param_range[1], name=param_name))
                
            elif isinstance(param_range[0], float) and isinstance(param_range[1], float):
                # Create a Real parameter range for floats
                param_space.append(Real(param_range[0], param_range[1], name=param_name))

        elif isinstance(param_range, range):
            # Convert range to min and max bounds
            param_space.append(Integer(min(param_range), max(param_range), name=param_name))
        else:
            raise ValueError(f"Invalid range for parameter '{param_name}': {param_range}")
    return param_space
# 1. Load historical data
def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=["date"], index_col="date")
    return data

# 2. Define a sample trading strategy
def trading_strategy(data, long_window, short_window):
    data["SMA_Short"] = data["Close"].rolling(window=short_window).mean()
    data["SMA_Long"] = data["Close"].rolling(window=long_window).mean()
    data["Signal"] = np.where(data["SMA_Short"] > data["SMA_Long"], 1, 0)  # 1 for Buy, 0 for Sell
    data["return"] = data["Close"].pct_change()
    data["Strategy_Return"] = data["Signal"].shift(1) * data["return"]
    data["creturns"] = (1 + data["Strategy_Return"]).cumprod() - 1  # Calculate cumulative returns
    return data

# 3. Split data into in-sample and out-of-sample periods
def split_data(data, train_size, test_size, step_size):
    start = 0
    while start + train_size + test_size <= len(data):
        train = data.iloc[start:start + train_size]
        test = data.iloc[start + train_size:start + train_size + test_size]
        yield train, test
        start += step_size


# 4. Optimize parameters on in-sample data
def optimize_parameters(train_data, param_grid):
    best_params = None
    best_performance = -np.inf
    for params in ParameterGrid(param_grid):
        result = trading_strategy(train_data.copy(), **params)
        performance = result['creturns'].iloc[-1]  # Get the last value of cumulative returns
        print(performance, best_performance, params)
        if performance > best_performance:
            best_performance = performance
            best_params = params
        
    print(f'In sample best performance: {best_performance}')
    return best_params

# 4. Optimize parameters on in-sample data using gp_minimize
def optimize_parameters_gp(train_data, param_space):

    if isinstance(param_space, dict):
        param_space = dict_to_param_space(param_space)

    @use_named_args(param_space)
    def objective(**params):
        result = trading_strategy(train_data.copy(), **params)
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

# Example Usage
# Load data
# 5. Test optimized parameters on out-of-sample data
def test_strategy(test_data, best_params):
    result = trading_strategy(test_data.copy(), **best_params)
    if "creturns" in result.columns:
        return result["creturns"].iloc[-1]
    else:
        return np.nan

# 6. Walk-forward optimization loop
def walk_forward_optimization(data, train_size, test_size, step_size, param_grid, optimize_fn):
    results = []
    for train, test in split_data(data, train_size, test_size, step_size):
        print(f"Training on data from {train.index[0]} to {train.index[-1]}")
        print(f"Testing on data from {test.index[0]} to {test.index[-1]}")
        
        # Optimize on training data
        best_params = optimize_fn(train, param_grid)
        print(f"Best params: {best_params}")
        
        # Test on out-of-sample data
        performance = test_strategy(test, best_params)
        print(f"Out-of-sample performance: {performance}")
        
        results.append(performance)
    return results