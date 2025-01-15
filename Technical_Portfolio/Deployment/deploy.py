import ccxt
import pandas as pd
import numpy as np
import os
import time
from unsync import unsync
import datetime as dt
import sys
from concurrent.futures import ThreadPoolExecutor
import re
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Management')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Portfolio_Optimization')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Strategies', 'Trend_Following')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Strategies', 'Mean_Reversion')))

# Import the modules
from data import Data, get_halal_symbols
from fetch_symbols import get_symbols
from sprtrnd_breakout import Sprtrnd_Breakout
from last_days_low import Last_Days_Low
from portfolio_management import Portfolio_Management
from portfolio_optimization import Portfolio_Optimization
from portfolio_risk_management import Portfolio_RM



class Deploy():
    def __init__(self, api_key, api_secret, exchange, halal_symbols, train_size, test_size, step_size, low_corr_thresh, market_data_filename, strategy_data_filename, timeframe, strategy_optimization_frequency, portfolio_optimization_frequency, portfolio_management_frequency):
        self.api_key = 'yqPWrtVuElaIExKmIp/E/upTOz/to1x7tC3JoFUxoSTKWCOorT6ifF/B'
        self.api_secret = 'L8h5vYoAu/jpQiBROA9yKN41FGwZAGGVF3nfrC5f5EiaoF7VksruPVdD7x1VOwnyyNCMdrGnT8lP4xHTiBrYMQ=='
        self.exchange = ccxt.kraken({
            'apiKey': api_key,
            'secret': api_secret
        })
        self.train_size = 440
        self.test_size = 200
        self.step_size = 200
        self.low_corr_thresh = 1.0
        self.market_data_filename = 'market_data.csv'
        self.strategy_data_filename = 'strategy_data.csv'
        self.timeframe = '1h'
        self.best_params = None
        self.best_weights = None
        data = self.load_data_from_csv(market_data_filename)
        strat_1_instance = Last_Days_Low(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size)
        strat_2_instance = Sprtrnd_Breakout(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size)
        self.halal_symbols = get_halal_symbols()
        cash_df = pd.DataFrame(data={'strategy': np.zeros(data.shape[0]), 'portfolio_value': np.ones(data.shape[0])}, index=data.index)
        self.strategy_map = {
            'cash_strat': cash_df,
            'strat_1': strat_1_instance,
            'strat_2': strat_2_instance
        }
        self.strategy_optimization_frequency = self.step_size
        self.portfolio_optimization_frequency = 300 #Every 2 Weeks
        self.portfolio_management_frequency = 4400 #Around 6 months
        self.counter = 0
        self.best_params = None
        self.best_weights = None
        self.symbols_to_liquidate = None
        self.selected_strategy = None
        self.drawdown_threshold = -0.15
        self.max_rows_market_data = self.market_data_size = 2000
        
    def symbols_in_current_balance(self):
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
        except ccxt.BaseError as e:
            print(f"An error occurred: {e}")
            
    def buy(self, to_add, coin):
        try:
            order = self.exchange.create_market_buy_order(coin, to_add)
            print(f"Buy order placed: {order}")
        except Exception as e:
            print(f"Error: {e}")
    def sell(self, to_sell, coin):
        try:
            order = exchange.create_market_sell_order(coin, to_sell)
            print(f"Sell order placed: {order}")
        except Exception as e:
            print(f"Error: {e}")
    def liquidate(self, symbols):
        try:
            # Step 1: Get your balances
            balance = self.exchange.fetch_balance()

            # Step 2: Loop through all assets in your balance and sell them
            for coin, coin_balance in balance['free'].items():
                if coin in symbols:
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
            
    def upload_complete_market_data(self, data_size = 2200):
        start_time = (dt.datetime.now() - dt.timedelta(hours= data_size)).date()
        end_time = dt.datetime.now().date()
        timeframes = ['1w', '1d', '4h', '1h', '30m','15m', '5m', '1m']
        index = 3 #It is better to choose the highest frequency for the backtest to be able to downsample
        interval = timeframes[index]
        data_instance = Data(halal_symbols, interval, start_time, end_time, exchange = 'kraken')
        data = data_instance.df
        last_date_data = data.index.get_level_values(0).unique()[-1].tz_localize('UTC')
        
        if dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) != last_date_data:
            time_difference = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) - last_date_data
            hours_difference = time_difference.total_seconds() / 3600 # Get the number of hours
            missing_data = fetch_latest_data(halal_symbols, interval, limit = int(hours_difference) + 1).result()
            complete_data = pd.concat([data, missing_data])
            
        complete_data.index = complete_data.index.set_levels(pd.to_datetime(complete_data.index.levels[0]), level=0)
        complete_data.to_csv('market_data.csv')
        print('Market data updated successfully')
        
    #Helper function
    def get_last_row(self, data):
        """Get the last date in the dataset."""
        last_date = data.index.get_level_values("date").max()
        last_date_data = data.loc[last_date]
        return last_date_data
    
    def get_portfolio_value(self, exchange):
        try:
            # Fetch account balances
            balances = exchange.fetch_balance()

            # Fetch tickers to get the latest prices
            tickers = exchange.fetch_tickers()

            # Calculate portfolio value in USD (or another base currency)
            portfolio_value = 0.0

            for currency, balance in balances['total'].items():
                if balance > 0:
                    if currency == "USD":
                        # Add USD cash directly to portfolio value
                        portfolio_value += balance
                    else:
                        # Use the USD pair or the most liquid market
                        pair = f"{currency}/USD"
                        if pair in tickers:
                            price = tickers[pair]['last']
                            portfolio_value += balance * price
                        else:
                            # Handle currencies without USD pairs (e.g., trade to BTC, then USD)
                            btc_pair = f"{currency}/BTC"
                            if btc_pair in tickers:
                                btc_price = tickers[btc_pair]['last']
                                usd_price = tickers["BTC/USD"]['last']
                                portfolio_value += balance * btc_price * usd_price

            return round(portfolio_value, 2)

        except ccxt.BaseError as e:
            print(f"An error occurred: {str(e)}")
            return None

        
    def format_symbols(self, symbols):
        """Converts the symbols to a format that the exchange understands."""
        if symbols[0].endswith('T'):
            symbols = [s[:-1] for s in symbols]
        formatted_symbols = [symbol.replace("USD", "/USD") for symbol in symbols]
        return formatted_symbols

    def filter_halal_df(self, data):
        # Drop multiple coins
        halal_symbols = ['BTC/USD', 'ETH/USD', 'LTC/USD']
        data_filtered = data[data.index.get_level_values("coin").isin(halal_symbols)]
        return data_filtered
    

    @unsync
    def fetch_latest_data(symbols, timeframe, limit=2):
        """Fetch latest OHLCV data for multiple symbols and stack them into a single DataFrame."""
        
        formatted_symbols = format_symbols(symbols)
        
        def fetch_symbol_data(symbol):
            """Fetch data for a single symbol and return a DataFrame."""
            try:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['coin'] = symbol
                return df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                return pd.DataFrame()  # Return an empty DataFrame if fetching fails

        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust workers based on CPU
            results = list(executor.map(fetch_symbol_data, formatted_symbols))

        # Concatenate all DataFrames and set multi-level index
        data_frames = [df for df in results if not df.empty]
        if data_frames:
            stacked_df = pd.concat(data_frames)
            stacked_df.set_index('coin', append=True, inplace=True)
            stacked_df = stacked_df[~stacked_df.index.duplicated()]  # Remove duplicates
            df = data_instance.prepare_data(stacked_df.unstack())
            df.reset_index(level = 1, inplace = True)
            df['coin'] = df['coin'].str.replace('/USD', 'USDT', regex=False)
            df.set_index('coin', append = True, inplace = True)
            return df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data
            
        # Append new data to CSV and maintain max length (asynchronous)
    @unsync
    def append_to_csv_with_limit(self, data):
        """_summary_

        Args:
            data (_type_): _description_
            filename (_type_): _description_
            max_rows (int, optional): _description_. Defaults to 2202. Should be account for the max number of rows needed for any of the processes
        """
        file_exists = os.path.isfile(filename)
        df = pd.DataFrame(data)
        
        if file_exists:
            existing_df = pd.read_csv(filename, index_col=['date', 'coin'], parse_dates=['date'])
            if existing_df.index.get_level_values(0).unique()[-1] == latest.index.get_level_values(0).unique()[-1]:
                return
            combined_df = pd.concat([existing_df, df])
            if len(combined_df) > max_rows:
                combined_df = combined_df.iloc[-max_rows:]  # Keep only the last max_rows rows
            combined_df.to_csv(filename)
        else:
            print('File does not exist')
            df.to_csv(filename, mode='w', header=True)
            
    #Getting the data from csv
    def load_data_from_csv(self, filename):
        if os.path.isfile(filename):
            data = pd.read_csv(filename, index_col=['date', 'coin'], parse_dates=['date'])
            if len(data) >= train_size + test_size:
                return data
        else:
            return pd.DataFrame()
    
    def perform_portfolio_rm(self, best_weights, exchange, drawdown_threshold = -0.15):
        
        current_strategy_returns_df = pd.read_csv('strategy_returns.csv')

        portfolio_returns = np.dot(best_weights, current_strategy_returns_df.T)

        portfolio_rm_instance = Portfolio_RM(portfolio_returns)

        drawdown_limit, in_drawdown = portfolio_rm_instance.drawdown_limit(drawdown_threshold)

        if in_drawdown.iloc[-1]:
            #Liquidate the portfolio
            print(f'Liquidating the portfolio because in_drawdown in {in_drawdown.iloc[-1]}')
            symbols_to_liquidate = self.symbols_in_current_balance(exchange)
            self.liquidate(symbols_to_liquidate, exchange)
            return True
        else :
            print(f'Portfolio is not in drawdown because in drawdown is {in_drawdown.iloc[-1]}')
            return False
        
    def run_wfo_and_get_results_returns(self, strategy_map):
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
                results_strategy_returns[key] = value.results.strategy
            elif key == 'cash_strat':
                results_strategy_returns[key] = value.strategy
                
        #Get the strategy returns df
        strategy_returns_df = pd.concat(results_strategy_returns, axis = 1).fillna(0)
        strategy_returns_df.to_csv('strategy_returns.csv')
        
        return results_strategy_returns
            
    def perform_portfolio_management(strategy_map, low_corr_threshold = 1):
        """_summary_

        Args:
            strategy_map (_type_): _description_
            low_corr_threshold (int, optional): _description_. Defaults to 1.
        """
        results_strategy_returns = run_wfo_and_get_results_returns(strategy_map)

        portfolio_management = Portfolio_Management(results_strategy_returns)

        keys_for_selected_strategy = portfolio_management.filter_by_correlation(low_corr_threshold=low_corr_threshold).columns

        selected_strategy = {key: value for key, value in strategy_map.items() if key in keys_for_selected_strategy}

        return selected_strategy
        
    def perform_optimization(self, strategy_map):
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
        
    def perform_portfolio_optimization(self, strategy_map, strategy_returns_df):
        """_summary_

        Args:
            strategy_returns_df (_type_): _description_
            train_size (int, optional): _description_. Defaults to 1000.
            test_size (int, optional): _description_. Defaults to 1000.
            step_size (int, optional): _description_. Defaults to 1000.
        """
        results_strategy_returns = self.run_wfo_and_get_results_returns(strategy_map)
        
        #Get portfolio optimization instance
        portfolio_optimization_instance = Portfolio_Optimization(log_rets = results_strategy_returns, train_size = train_size, test_size = test_size, step_size = step_size, objective = 'multiple')

        #Run the optimization
        train_data = strategy_returns_df.iloc[-train_size:]
        best_weights = portfolio_optimization_instance.optimize_weights_minimize(train_data)

        return best_weights
    
    
    def run_strategy(self, exchange, halal_symbols, selected_strategy, best_params, best_weights, timeframe = '1h'):
        #Get the current_total_balance
        current_total_balance = get_portfolio_value(exchange)

        #Store the max allocation for each strategy in a dictionary
        max_allocation_map = {
            key: best_weights[i] * current_total_balance / strategy.max_universe
            for i, (key, strategy) in enumerate(selected_strategy.items())
            if i < len(best_weights) and best_weights[i] > 0 and key != 'cash_strat'
        }

        #Rebuild the strategy map, with the updated max_allocation for each strategy
        for key, value in selected_strategy.items():
            if key != 'cash_strat':
                value.max_dollar_allocation = max_allocation_map[key]
                
        
        timeframe = timeframe
        latest = fetch_latest_data(halal_symbols, timeframe).result()
        append_to_csv_with_limit(latest, 'market_data.csv').result()
        data = load_data_from_csv('market_data.csv')
        
        
        #Run each strategy on enough data points and get the total portfolio value
        length_of_data_to_run_strategy = 100 #This does not have to do with anything with test_size or train_size,
            #but it is better to be equal to the test_size because we want get the latest returns of the strategy with the latest best_params to get the current strategy returns
        data_to_run_strategy = data.iloc[-length_of_data_to_run_strategy:]
        
        current_strategy_results = {
            key: value.trading_strategy(data_to_run_strategy, best_params[key])
            for key, value in selected_strategy.items()
            if key != 'cash_strat'
        }
        
        current_allocation_strategy_map = {
            key: value['current_allocation']
            for key, value in current_strategy_results.items()
            if key != 'cash_strat'
        }
        
        current_allocation_results_df = pd.concat(current_allocation_strategy_map, axis=1).fillna(0).sum(axis=1)
        
        current_allocation = get_last_row(current_allocation_results_df)
        
        current_universe = list(
            set(
                value.current_universe
                for value in selected_strategy.values()
                if key != 'cash_strat'
            )
        )

        symbols_in_current_balance = symbols_in_current_balance(exchange)
        # Find symbols in current balance but not in current universe
        symbols_not_in_universe = [symbol for symbol in symbols_in_current_balance if symbol not in current_universe]

        # Liquidate the symbols not in the current universe
        print(f"Liquidating {symbols_not_in_universe}...")
        liquidate(symbols_not_in_universe, exchange)
        print("Liquidation complete.")

        for coin in current_universe:
            formatted_coin = coin.replace('USDT', '')
            coin_balance = get_coin_balance(formatted_coin)
            current_coin_allocation = get_coin_allocation(coin, current_allocation)
            
            to_add = current_coin_allocation - coin_balance
            
            if to_add > 0:
                print(f"Adding {to_add} {formatted_coin} to the portfolio...")
                buy(to_add, coin, exchange)
            elif to_add < 0:
                print(f"Selling {-to_add} {formatted_coin} from the portfolio...")
                sell(-to_add, coin, exchange)
                
        # This will be used to plot the current portfolio 
        portfolio_returns = np.dot(best_weights, current_strategy_returns_df.T)
        portfolio_cumulative_returns = portfolio_returns.cumsum().apply(np.exp)
        portfolio_cumulative_returns.plot()
        
    def main_loop(self):
        # THE MAIN LOOP
        while True:
            now = dt.datetime.now()
            next_hour = (now + dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
            sleep_duration = (next_hour - now).total_seconds()
            time.sleep(sleep_duration)
            data = self.load_data_from_csv(self.market_data_filename)

            if counter % self.strategy_optimization_frequency == 0:
                best_params = self.perform_optimization(self.strategy_map)

            if counter % self.portfolio_optimization_frequency == 0:
                best_weights = self.perform_portfolio_optimization(self.strategy_map, train_size=self.train_size, test_size=test_size, step_size=step_size)

            if counter % self.portfolio_management_frequency == 0:
                selected_strategy = perform_portfolio_management(strategy_map, low_corr_threshold=low_corr_thresh)

            if self.perform_portfolio_rm(best_weights, exchange, drawdown_threshold=drawdown_threshold):
                continue

            self.frun_strategy(exchange, halal_symbols, selected_strategy, best_params, best_weights, timeframe=timeframe)
            
            self.counter += 1
        
    