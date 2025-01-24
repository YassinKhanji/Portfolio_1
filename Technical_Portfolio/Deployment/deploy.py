import ccxt
import pandas as pd
import numpy as np
import os
import time
from unsync import unsync
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Management')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Portfolio_Optimization')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Strategies', 'Trend_Following')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Strategies', 'Mean_Reversion')))

# Import the modules
from data import Data, get_symbols_for_bot
from sprtrnd_breakout import Sprtrnd_Breakout
from last_days_low import Last_Days_Low
from portfolio_management import Portfolio_Management
from portfolio_optimization import Portfolio_Optimization
from portfolio_risk_management import Portfolio_RM


api_key = 'yqPWrtVuElaIExKmIp/E/upTOz/to1x7tC3JoFUxoSTKWCOorT6ifF/B'
api_secret = 'L8h5vYoAu/jpQiBROA9yKN41FGwZAGGVF3nfrC5f5EiaoF7VksruPVdD7x1VOwnyyNCMdrGnT8lP4xHTiBrYMQ=='
exchange = ccxt.kraken({
    'apiKey': api_key,
    'secret': api_secret,
    'options': {
        'defaultType': 'spot',  # Ensure only spot markets are considered
    }
})
train_size = 500
test_size = 500
step_size = 500
low_corr_thresh = 1.0
strategy_optimization_frequency = step_size
portfolio_optimization_frequency = 300 #Every 2 Weeks
portfolio_management_frequency = 4400 #Around 6 months
counter = 0
best_params = None
best_weights = None
symbols_to_liquidate = None
selected_strategy = None
live_selected_strategy = None
data_instance = None
results_strategy_returns_ = None
drawdown_threshold = -0.15
max_rows_market_data = market_data_size = 2000
length_of_data_to_run_strategy = 500
_symbols_threshold = 750 #Get new symbols every month
market_data_filename = 'market_data.csv'
strategy_data_filename = 'strategy_returns.csv'
portfolio_returns_filename = "portfolio_returns.csv"
timeframe = '1h'
symbols_to_trade = get_symbols_for_bot()
for symbol in ['XRPUSD', 'ETHUSD', 'BTCUSD']:
    if symbol not in symbols_to_trade:
        symbols_to_trade.append(symbol)
# symbols_to_trade = ['BTCUSD', 'ETHUSD', 'XRPUSD', 'SOLUSD', 'BONKUSD']
# symbols_to_trade = ['STORJUSD', 'MINAUSD', 'TRXUSD', 'RADUSD', 'TAOUSD', 'OGNUSD', 'IMXUSD', 'ZKUSD', 'FILUSD', 'MASKUSD', 'FORTHUSD', 'LSKUSD', 'MANAUSD', 'ADAUSD', 'FXSUSD', 'TONUSD', 'AVAXUSD', 'GMTUSD', 'SAGAUSD', 'SEIUSD', 'DOTUSD', 'ETCUSD', 'BLURUSD', 'ANKRUSD', 'WIFUSD']


def format_symbols(symbols):
    """Converts the symbols to a format that the exchange understands."""
    if symbols[0].endswith('T'):
        symbols = [s[:-1] for s in symbols]
    return [symbol.replace("USD", "/USD") for symbol in symbols]

def upload_complete_market_data():
    start_time = (dt.datetime.now() - dt.timedelta(hours= max_rows_market_data)).date()
    end_time = dt.datetime.now().date()
    timeframes = ['1w', '1d', '4h', '1h', '30m','15m', '5m', '1m']
    index = 3 #It is better to choose the highest frequency for the backtest to be able to downsample
    interval = timeframes[index]
    data_instance = Data(symbols_to_trade, interval, start_time, end_time, exchange = 'kraken')
    data = data_instance.df
    
    complete_missing_data(data_instance, data)

def complete_missing_data(data_instance, data):
    last_date_data = data.index.get_level_values(0).unique()[-1].tz_localize('UTC')
    if dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours = 1) != last_date_data:
        time_difference = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) - dt.timedelta(hours = 1) - last_date_data
        hours_difference = time_difference.total_seconds() / 3600 # Get the number of hours
        if hours_difference < 3:
            print(f'Changing the hours difference to 3 hours')
            hours_difference = 3
        missing_data = fetch_latest_data(data_instance, limit = int(hours_difference) + 1)
        complete_data = pd.concat([data, missing_data])
        complete_data.index = complete_data.index.set_levels(pd.to_datetime(complete_data.index.levels[0]), level=0)
        complete_data = complete_data[~complete_data.index.duplicated(keep='last')]
        complete_data.to_csv(market_data_filename)
        print('Market data updated successfully')
    else:
        print('No missing data')
        data.to_csv(market_data_filename)
        print('Market data updated successfully')

def fetch_latest_data(data_instance, limit=3):
    """Fetch latest OHLCV data for multiple symbols and stack them into a single DataFrame."""
    
    formatted_symbols = format_symbols(symbols_to_trade)
    
    def fetch_symbol_data(symbol, formatted_symbol):
        """Fetch data for a single symbol and return a DataFrame."""
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['coin'] = formatted_symbol
            return df[:-1]
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            try:
                # Retry fetching data
                ohlcv = exchange.fetch_ohlcv(formatted_symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['coin'] = formatted_symbol
                return df[:-1]
            except Exception as e:
                print(f"Error fetching data for {symbol} on retry: {e}")
                return pd.DataFrame()

    # Use ThreadPoolExecutor for parallel requests
    with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust workers based on CPU
        results = list(executor.map(fetch_symbol_data, symbols_to_trade, formatted_symbols))

    # Concatenate all DataFrames and set multi-level index
    data_frames = [df for df in results if not df.empty]
    if data_frames:
        stacked_df = pd.concat(data_frames)
        stacked_df.set_index('coin', append=True, inplace=True)
        stacked_df = stacked_df[~stacked_df.index.duplicated()]  # Remove duplicates
        df = data_instance.prepare_data(stacked_df.unstack())
        df.reset_index(level = 1, inplace = True)
        df['coin'] = df['coin'].str.replace('/USD', 'USDT', regex=False).replace('USD', 'USDT', regex=False)
        df.set_index('coin', append = True, inplace = True)
        return df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data

        
def load_data_from_csv():
    filename = market_data_filename
    if os.path.isfile(filename):
        try:
            data = pd.read_csv(filename, index_col=[0, 1], parse_dates=[0])
            if len(data.unstack()) >= train_size + test_size:
                print(f'Returning data. Its size: {len(data.unstack())}')
                return data
            else:
                print(f'Data not large enough. Its size: {len(data.unstack())}')
                return
        except Exception as e:
            print(f'File does not exist or is empty: {e}')
    else:
        print(f'The file is empty or does not exist')
        
def get_portfolio_value():
    try:
        #Opening a fresh new session for the exchange
        exchange = ccxt.kraken({
        'apiKey': api_key,
        'secret': api_secret,
        'options': {
            'defaultType': 'spot',  # Ensure only spot markets are considered
            }
        })
        
        # Fetch account balances
        balances = exchange.fetch_balance()
        # Fetch tickers to get the latest prices
        tickers = exchange.fetch_tickers()
        # Initialize portfolio value
        portfolio_value = 0.0

        for currency, balance in balances['total'].items():
            if balance > 0:
                if currency == "USD":
                    portfolio_value += balance
                else:
                    pair = f"{currency}/USD"
                    if pair in exchange.markets:
                        market = exchange.markets[pair]
                        if market['type'] == 'spot':  # Ensure it's a spot market
                            price = tickers[pair]['last']
                            portfolio_value += balance * price

        return round(portfolio_value, 2)

    except ccxt.BaseError as e:
        print(f"An error occurred: {str(e)}")
        return None
    
    
current_total_balance = get_portfolio_value() #Testing kraken api connection
print(f"Current Total Balance: {current_total_balance}")
print(f"Uploading Data First for {len(symbols_to_trade)} symbols: {symbols_to_trade}")

################### Uploading Data ###################
start_time = (dt.datetime.now() - dt.timedelta(hours= max_rows_market_data)).date()
end_time = dt.datetime.now().date()
timeframes = ['1w', '1d', '4h', '1h', '30m','15m', '5m', '1m']
index = 3 #It is better to choose the highest frequency for the backtest to be able to downsample
interval = timeframes[index]
data_instance = Data(symbols_to_trade, interval, start_time, end_time, exchange = 'kraken')
data = data_instance.df

complete_missing_data(data_instance, data)
#######################################################
print('Data Uploaded, Now Loading Data')
data = load_data_from_csv()
print('Data Loaded')

strat_1_instance = Last_Days_Low(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size)
strat_2_instance = Sprtrnd_Breakout(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size)
live_strat_1_instance = Last_Days_Low(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size, live = True)
live_strat_2_instance = Sprtrnd_Breakout(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size, live = True)
cash_df = pd.DataFrame(data={'strategy': np.zeros(data.shape[0]), 'portfolio_value': np.ones(data.shape[0])}, index=data.index)
strategy_map = {
    # 'cash_strat': cash_df,
    'strat_1': strat_1_instance,
    'strat_2': strat_2_instance
}
live_strategy_map = {
    # 'cash_strat': cash_df,
    'strat_1': live_strat_1_instance,
    'strat_2': live_strat_2_instance
}

############ Helper Methods ############
def symbols_in_current_balance():
    # Fetch account balance
    try:
        balance = exchange.fetch_balance()

        # Extract symbols with non-zero balance
        symbols = [
            f'{currency}USDT'
            for currency, info in balance['total'].items()
            if info > 0
        ]

        print("Symbols in your current balance:", symbols)
        return symbols
    except ccxt.BaseError as e:
        print(f"An error occurred: {e}")
        

def get_coin_balance(formatted_coin):
    try:
        balance = exchange.fetch_balance()
        coin_balance= balance['total'][formatted_coin]
        if coin_balance is not None:
            return coin_balance
        else:
            return 0
    except Exception as e:
        print(f"Error fetching balance for {formatted_coin}: {e}")
        return None
    

def get_usd_left():
    return exchange.fetch_balance()['free']['USD']
        
def buy(to_add, coin):
    try:
        
        order = exchange.create_market_buy_order(coin, to_add)
        print(f"Buy order placed: {order}")
    except Exception as e:
        print(f"Error: {e}")
        
def sell(to_sell, coin):
    try:
        order = exchange.create_market_sell_order(coin, to_sell)
        print(f"Sell order placed: {order}")
    except Exception as e:
        print(f"Error: {e}")
        
def liquidate(symbols_to_liquidate):
    try:
        # Step 1: Get your balances
        balance = exchange.fetch_balance()
        cant_liquidate = ['USD', 'CAD']

        # Step 2: Loop through all assets in your balance and sell them
        for coin, coin_balance in balance['free'].items():
            if coin in symbols_to_liquidate and coin not in cant_liquidate:
                if coin_balance > 0:  # Only sell if you have a non-zero balance
                    print(f"Selling {coin_balance} {coin}...")

                    # Determine the symbol for the sell order (e.g., BTC/USD, ETH/USDT)
                    symbol = f"{coin}/USD"  # Replace USD with your preferred quote currency
                    order = exchange.create_market_sell_order(symbol, coin_balance)
                    print(f"Sell order placed: {order}")
                else:
                    print(f"No {coin} to sell.")

        print("All possible assets have been liquidated.")

    except Exception as e:
        print(f"Error: {e}")
        
def perform_portfolio_rm():
    if os.path.isfile(portfolio_returns_filename) and os.path.getsize(portfolio_returns_filename) > 0:
        try:
            current_portfolio_returns_df = pd.read_csv(
                strategy_data_filename,
                index_col=[0],
                parse_dates=[0]
            )
        except pd.errors.EmptyDataError:
            print("The file is empty or has no valid data.")
            return False
    else:
        print(f"File {portfolio_returns_filename} does not exist or is empty.")
        return False


    if current_portfolio_returns_df.empty or len(current_portfolio_returns_df) < train_size + test_size:
        return False
    
    portfolio_returns_series = pd.Series(current_portfolio_returns_df)
    
    portfolio_rm_instance = Portfolio_RM(portfolio_returns_series)

    drawdown_limit, in_drawdown = portfolio_rm_instance.drawdown_limit(drawdown_threshold)

    if in_drawdown.iloc[-1]:
        #Liquidate the portfolio
        print(f'Liquidating the portfolio because in_drawdown in {in_drawdown.iloc[-1]}')
        symbols_to_liquidate = symbols_in_current_balance()
        symbols_to_liquidate = [s.replace('USDT', '') for s in symbols_to_liquidate]
        liquidate(symbols_to_liquidate)
        return True
    else :
        print(f'Portfolio is not in drawdown because in drawdown is {in_drawdown.iloc[-1]}')
        return False

def run_wfo_and_get_results_returns():
    """_summary_
    Takes the strategy map, runs the WFO for each strategy and returns the results of the strategy returns after the WFO.
    It also adds the df of the strategy returns to a csv file

    Args:
        strategy_map (_type_): _description_

    Returns:
        results_strategy_returns (_type_): _description_ the results of the strategy returns after the WFO
    """
    #Run the WFO for each strategy (but the cash strategy)
    for key, value in strategy_map.items():
        if key != 'cash_strat':
            value.test()
            
    #Make a new dictionary that contains the results strategy returns of the WFO
    results_strategy_returns = {}
    for key, value in strategy_map.items():
        if key != 'cash_strat':
            results_strategy_returns[key] = value.results.strategy.reindex(cash_df.index).fillna(0)
        elif key == 'cash_strat':
            results_strategy_returns[key] = value.strategy
    
    return results_strategy_returns
def perform_portfolio_management(results_strategy_returns):
    """_summary_

    Args:
        strategy_map (_type_): _description_
        low_corr_threshold (int, optional): _description_. Defaults to 1.
    """
    if counter % portfolio_optimization_frequency == 0:
        print('Already have the results strategy returns from the portfolio optimization, Skipping the WFO Process...')
    else:
        results_strategy_returns = run_wfo_and_get_results_returns()

    portfolio_management = Portfolio_Management(results_strategy_returns)

    keys_for_selected_strategy = portfolio_management.filter_by_correlation(low_corr_threshold= low_corr_thresh).columns

    selected_strategy = {key: value for key, value in strategy_map.items() if key in keys_for_selected_strategy}
    
    live_selected_strategy = {key: value for key, value in live_strategy_map.items() if key in keys_for_selected_strategy and key != 'cash_strat'}
    
    return selected_strategy, live_selected_strategy
    
def perform_optimization():
    """_summary_

    Args:
        strategy_map (_type_): _description_
    """

    #Run the optimization to get the strategy parameters
    for key, value in strategy_map.items():
        if key != 'cash_strat':
            value.optimize()

    #Storing the best_params for each strategy in a separate dictionary
    best_params = {key: value.best_params for key, value in strategy_map.items() if key != 'cash_strat'}
    
    return best_params
    
def perform_portfolio_optimization():
    """_summary_

    Args:
        strategy_returns_df (_type_): _description_
        train_size (int, optional): _description_. Defaults to 1000.
        test_size (int, optional): _description_. Defaults to 1000.
        step_size (int, optional): _description_. Defaults to 1000.
    """
    results_strategy_returns = run_wfo_and_get_results_returns()
    
    #Get portfolio optimization instance
    portfolio_optimization_instance = Portfolio_Optimization(log_rets = results_strategy_returns, train_size = train_size, test_size = test_size, step_size = step_size, objective = 'multiple')

    #Run the 
    results_strategy_returns_df = pd.concat(results_strategy_returns, axis = 1).fillna(0)
    train_data = results_strategy_returns_df.iloc[-train_size:]
    best_weights = portfolio_optimization_instance.optimize_weights_minimize(train_data)
    
    return best_weights, results_strategy_returns


def run_strategy(best_params, best_weights, live_selected_strategy, in_drawdown):
    #Get the current_total_balance
    current_total_balance = get_portfolio_value()
    
    print(f"Best Weights: {best_weights}")
    print(f"Current Total Balance: {current_total_balance}")
    print(f"Live Selected Strategy: {live_selected_strategy}")
    print(f'Best Params: {best_params}')
    
    
    ################### Max Allocation ###################
    #Store the max allocation for each strategy in a dictionary
    max_allocation_map = {
        key: best_weights[i] * current_total_balance / strategy.max_universe
        for i, (key, strategy) in enumerate(live_selected_strategy.items())
        if i < len(best_weights) and best_weights[i] > 0 and key != 'cash_strat'
    }

    print(f'Max_allocation_map: {max_allocation_map}')
    #Rebuild the strategy map, with the updated max_allocation for each strategy
    for key, value in live_selected_strategy.items():
        if key != 'cash_strat':
            value.max_dollar_allocation = max_allocation_map.get(key, 0)
            print(f"Max Dollar Allocation for {key}: {value.max_dollar_allocation}")
    
    ####################### Preparing Data #######################
    print('Loading Data...')
    data = load_data_from_csv()
    print('Data Loaded: ', data)
    print('Completing missing data...')
    complete_missing_data(data_instance, data)
    
    print('Data Uploaded, Now Loading Final Data.')
    data = load_data_from_csv()
    
    
    #Run each strategy on enough data points and get the total portfolio value
    data_to_run_strategy = data.unstack().iloc[-length_of_data_to_run_strategy:].stack(future_stack = True)
    print(f'Data to run the strategy on: {data_to_run_strategy}')
    
    
    ################### Running Strategy on Data ###################
    current_strategy_results = {
        key: value.trading_strategy(data_to_run_strategy, best_params[key])
        for key, value in live_selected_strategy.items()
        if key != 'cash_strat'
    }

    for key, value in current_strategy_results.items():
        if 'strategy' in value.columns:
            print(f'Strategy Column is in {key}. Its strategy column tail: {value["strategy"].tail()}')
            print(f'Its position column tail: {value["position"].tail()}')
        else:
            print(f'Strategy not in columns. All other columns for {key}: {value.head()}')
            
    
    
    ################### Strategy Returns ###################
    current_strategy_returns = {
        key: value['strategy']
        for key, value in current_strategy_results.items()
    }

    # Ensure alignment with `data_to_run_strategy` index
    for key, value in current_strategy_returns.items():
        # Sum returns for the same datetime (level=0)
        summed_value = value.groupby(level=0).sum()
        print(f"Summed value for {key}: {summed_value}")

        # Step 2: Reindex to align with `data_to_run_strategy` index
        aligned_value = summed_value.reindex(data_to_run_strategy.index.get_level_values(0), level=0).fillna(0)

        # Assign the aligned value back
        current_strategy_returns[key] = aligned_value
        print(f"Strategy returns for {key}: {aligned_value}")
    
    # Concatenate and sum by the first index (datetime)
    current_strategy_returns_df = pd.concat(current_strategy_returns, axis=1).fillna(0)
    # cash_strategy = cash_df['strategy'].reindex(current_strategy_returns_df.index).dropna()
    # current_strategy_returns_df = pd.concat([current_strategy_returns_df, cash_strategy], axis=1).fillna(0)
    
    print(f'Current Strategy returns df: {current_strategy_returns_df}')
    print('Appending to Strategy returns data...')
    current_strategy_returns_df = current_strategy_returns_df[~current_strategy_returns_df.index.duplicated(keep='last')]
    if not os.path.isfile(strategy_data_filename) or os.path.getsize(strategy_data_filename) == 0:
        current_strategy_returns_df.to_csv(strategy_data_filename)
    else:
        current_strategy_returns_df[-1:].to_csv(strategy_data_filename, mode='a', header=False, index = True, date_format='%Y-%m-%d %H:%M:%S')
    print(f'Appending Done.')
    
    
    ################### Portfolio Returns ###################
    current_portfolio_returns = current_strategy_returns_df.dot(best_weights)
    print(f'Current Portfolio Returns: {current_portfolio_returns}')
    if not os.path.isfile(portfolio_returns_filename) or os.path.getsize(portfolio_returns_filename) == 0:
        current_portfolio_returns.to_csv(portfolio_returns_filename)
    else:
        current_portfolio_returns[-1:].to_csv(portfolio_returns_filename, mode='a', header=False, index = True, date_format='%Y-%m-%d %H:%M:%S')
    
    #################### Checking in Drawdown ################
    
    if in_drawdown:
        print('In Drawdown, Skipping Current Allocations, Universe, and Order placement...')
        return
    else:
        print('Not in Drawdown, Proceeding with Current Allocations, Universe, and Order placement...')
    
    ################### Current Allocations ###################
    #Getting the allocation
    current_allocation_strategy_map = {
        key: value['coin_amount_to_bought']
        for key, value in current_strategy_results.items()
        if key != 'cash_strat'
    }
    
    for key, value in current_allocation_strategy_map.items():
        print(f'Current Allocation for {key}: {value}')
    
    current_allocation_results_df = pd.concat(current_allocation_strategy_map, axis=1).fillna(0).sum(axis=1).sort_index()
    print(f'Current allocation results df: {current_allocation_results_df}')
    if not current_allocation_results_df.empty:
        last_index = current_allocation_results_df.index.get_level_values(0).unique()[-1] 
        print(f'Last index of current allocation results df: {last_index}')       
        current_allocation = current_allocation_results_df.loc[last_index]
        print(f'Current Allocations: {current_allocation}')
    else:
        print(f'Current allocation results df is empty')

    ######################### Universe ##########################
    # Extract current universes from selected_strategy
    print('Getting Universe')
    current_universes = [
        set(value.current_universe)  # Convert each universe to a set for comparison
        for key, value in live_selected_strategy.items()
        if key != 'cash_strat'
    ]
    print(f'Current Universes : {current_universes}')

    # Remove overlaps between universes
    # Start with the first set and iteratively remove overlaps
    unique_universes = []
    for universe in current_universes:
        for other_universe in unique_universes:
            universe -= other_universe  # Remove overlapping strings
        unique_universes.append(universe)

    # Convert sets back to lists (if needed)
    unique_universes = [list(universe) for universe in unique_universes]

    flattened_universe = [item for sublist in unique_universes for item in sublist]
    print(f'Current Universe: {flattened_universe}')


    symbols_in_current_portfolio = symbols_in_current_balance()
    print(f'Symbols in Current balance: {symbols_in_current_portfolio}')
    
    ################### Order Placement ###################
    # Ensure symbols_in_current_portfolio is not None
    if symbols_in_current_portfolio:
        symbols_not_in_universe = [
            symbol.replace('USDT', '').replace('USD', '') for symbol in symbols_in_current_portfolio
            if symbol not in flattened_universe
        ]
        print(f"Liquidating {symbols_not_in_universe}...")
        liquidate(symbols_not_in_universe)
        print("Liquidation complete.")
    else:
        print("symbols_in_current_portfolio is None or empty.")

    print(f'Current_universe: {flattened_universe}')
    for coin in flattened_universe:
        formatted_coin = coin.replace('USDT', '').replace('USD', '')
        coin_for_order = coin.replace('USDT', '/USD')
        coin_balance = get_coin_balance(formatted_coin)
        current_coin_allocation = current_allocation[coin]
        
        if coin_balance is None:
            coin_balance = 0
        
        print(f'Current coin allocation: {current_coin_allocation}')
        print(f'Coin balance: {coin_balance}')
        to_add = round(current_coin_allocation - coin_balance, 7)
        
        
        if to_add > 0 and to_add < get_usd_left():
            print(f"Adding {to_add} {formatted_coin} to the portfolio...")
            buy(to_add, coin_for_order)
        elif to_add < 0 and coin_balance >= abs(to_add):
            print(f"Selling {-to_add} {formatted_coin} from the portfolio...")
            sell(-to_add, coin_for_order)
        else:
            print(f"Nothing to add because {to_add} in coin's currency is almost $0.0")
            
            
def main_loop():
    # THE MAIN LOOP
    counter = 0
    while True:

        if counter % strategy_optimization_frequency == 0:
            print('Performing optimization')
            best_params = perform_optimization()

        if counter % portfolio_optimization_frequency == 0:
            print('Performing portfolio optimization')
            best_weights, results_strategy_returns = perform_portfolio_optimization()

        if counter % portfolio_management_frequency == 0:
            print('Performing portfolio management')
            selected_strategy, live_selected_strategy = perform_portfolio_management(results_strategy_returns)
        
        print('Adding to counter')
        counter += 1
        
        data = load_data_from_csv()
        print('Data Loaded: ', data)
        complete_missing_data(data_instance, data)
        print('Updating Data Before Portfolio RM')

        in_dradown = perform_portfolio_rm()
        
        if in_dradown:
            print('Performed portfolio risk management, portfolio is in drawdown')
        
                
        #Perform the strategy after each hour
        now = dt.datetime.now()
        print('Current time: ', now)
        next_hour = (now + dt.timedelta(hours=1)).replace(minute = 0, second=0, microsecond=0)
        print('Next hour: ', next_hour)
        sleep_duration = (next_hour - now).total_seconds()
        print('Sleep duration: ', sleep_duration)
        time.sleep(sleep_duration)
        
        print('Running strategy')
        print(f'Best Params: {best_params}')
        run_strategy(best_params, best_weights, live_selected_strategy, in_dradown)

        
        
# main_loop()