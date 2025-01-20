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



class Deploy():
    def __init__(self, train_size = 500, test_size = 500, step_size = 500):
        self.api_key = 'yqPWrtVuElaIExKmIp/E/upTOz/to1x7tC3JoFUxoSTKWCOorT6ifF/B'
        self.api_secret = 'L8h5vYoAu/jpQiBROA9yKN41FGwZAGGVF3nfrC5f5EiaoF7VksruPVdD7x1VOwnyyNCMdrGnT8lP4xHTiBrYMQ=='
        self.exchange = ccxt.kraken({
            'apiKey': self.api_key,
            'secret': self.api_secret,
            'options': {
                'defaultType': 'spot',  # Ensure only spot markets are considered
            }
        })
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.low_corr_thresh = 1.0
        self.strategy_optimization_frequency = self.step_size
        self.portfolio_optimization_frequency = 300 #Every 2 Weeks
        self.portfolio_management_frequency = 4400 #Around 6 months
        self.counter = 0
        self.best_params = None
        self.best_weights = None
        self.symbols_to_liquidate = None
        self.selected_strategy = None
        self.live_selected_strategy = None
        self.data_instance = None
        self.results_strategy_returns_ = None
        self.drawdown_threshold = -0.15
        self.max_rows_market_data = self.market_data_size = 2000
        self.length_of_data_to_run_strategy = 500
        reset_symbols_threshold = 750 #Get new symbols every month
        self.market_data_filename = 'market_data.csv'
        self.strategy_data_filename = 'strategy_returns.csv'
        self.timeframe = '1h'
        self.symbols_to_trade = get_symbols_for_bot()[:25]
        # self.symbols_to_trade = ['POLUSD']
        # self.symbols_to_trade = ['FORTHUSD', 'FTMUSD', 'QTUMUSD', 'LTCUSD', 'MEMEUSD', 'PEPEUSD', 'ETHUSD', 'BTCUSD', 'CVCUSD', 'PHAUSD', 'BCHUSD', 'OMNIUSD', 'ZKUSD', 'SHIBUSD', 'POLUSD', 'ARBUSD', 'PONDUSD', 'XRPUSD', 'ZROUSD', 'LPTUSD', 'SANDUSD', 'SYNUSD', 'GLMRUSD', 'ALTUSD', 'ENJUSD', 'MOVRUSD']
        current_total_balance = self.get_portfolio_value()
        print(f"Current Total Balance: {current_total_balance}")
        print(f"Uploading Data First for {len(self.symbols_to_trade)} symbols: {self.symbols_to_trade}")
        self.upload_complete_market_data()
        print('Data Uploaded, Now Loading Data')
        data = self.load_data_from_csv()
        self.data = data
        print('Data Loaded')
        strat_1_instance = Last_Days_Low(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size)
        strat_2_instance = Sprtrnd_Breakout(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size)
        live_strat_1_instance = Last_Days_Low(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size, live = True)
        live_strat_2_instance = Sprtrnd_Breakout(data, objective='multiple', train_size=train_size, test_size=test_size, step_size=step_size, live = True)
        self.cash_df = pd.DataFrame(data={'strategy': np.zeros(data.shape[0]), 'portfolio_value': np.ones(data.shape[0])}, index=data.index)
        self.strategy_map = {
            'cash_strat': self.cash_df,
            'strat_1': strat_1_instance,
            'strat_2': strat_2_instance
        }
        self.live_strategy_map = {
            'cash_strat': self.cash_df,
            'strat_1': live_strat_1_instance,
            'strat_2': live_strat_2_instance
        }
            
        
    ############ Helper Methods ############
    def symbols_in_current_balance(self):
        # Fetch account balance
        try:
            balance = self.exchange.fetch_balance()

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
            
    
    def get_coin_balance(self, formatted_coin):
        try:
            balance = self.exchange.fetch_balance()
            coin_balance= balance['total'][formatted_coin]
            if coin_balance is not None:
                return coin_balance
            else:
                return 0
        except Exception as e:
            print(f"Error fetching balance for {formatted_coin}: {e}")
            return None
        
    
    def get_usd_left(self):
        return self.exchange.fetch_balance()['free']['USD']
            
    def buy(self, to_add, coin):
        try:
            
            order = self.exchange.create_market_buy_order(coin, to_add)
            print(f"Buy order placed: {order}")
        except Exception as e:
            print(f"Error: {e}")
            
    def sell(self, to_sell, coin):
        try:
            order = self.exchange.create_market_sell_order(coin, to_sell)
            print(f"Sell order placed: {order}")
        except Exception as e:
            print(f"Error: {e}")
            
    def liquidate(self, symbols_to_liquidate):
        try:
            # Step 1: Get your balances
            balance = self.exchange.fetch_balance()
            cant_liquidate = ['USD', 'CAD']

            # Step 2: Loop through all assets in your balance and sell them
            for coin, coin_balance in balance['free'].items():
                if coin in symbols_to_liquidate and coin not in cant_liquidate:
                    if coin_balance > 0:  # Only sell if you have a non-zero balance
                        print(f"Selling {coin_balance} {coin}...")

                        # Determine the symbol for the sell order (e.g., BTC/USD, ETH/USDT)
                        symbol = f"{coin}/USD"  # Replace USD with your preferred quote currency
                        order = self.exchange.create_market_sell_order(symbol, coin_balance)
                        print(f"Sell order placed: {order}")
                    else:
                        print(f"No {coin} to sell.")

            print("All possible assets have been liquidated.")

        except Exception as e:
            print(f"Error: {e}")
            
    def get_portfolio_value(self):
        try:
            # Fetch account balances
            balances = self.exchange.fetch_balance()
            # Fetch tickers to get the latest prices
            tickers = self.exchange.fetch_tickers()
            # Initialize portfolio value
            portfolio_value = 0.0

            for currency, balance in balances['total'].items():
                if balance > 0:
                    if currency == "USD":
                        portfolio_value += balance
                    else:
                        pair = f"{currency}/USD"
                        if pair in self.exchange.markets:
                            market = self.exchange.markets[pair]
                            if market['type'] == 'spot':  # Ensure it's a spot market
                                price = tickers[pair]['last']
                                portfolio_value += balance * price

            return round(portfolio_value, 2)

        except ccxt.BaseError as e:
            print(f"An error occurred: {str(e)}")
            return None
        
    def format_symbols(self, symbols):
        """Converts the symbols to a format that the exchange understands."""
        if symbols[0].endswith('T'):
            symbols = [s[:-1] for s in symbols]
        return [symbol.replace("USD", "/USD") for symbol in symbols]

    def filter_halal_df(self, data):
        return data[data.index.get_level_values("coin").isin(self.symbols_to_trade)]
    
    def complete_missing_data(self, data):
        last_date_data = data.index.get_level_values(0).unique()[-1].tz_localize('UTC')
        if dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) != last_date_data:
            time_difference = dt.datetime.now(dt.UTC).replace(minute=0, second=0, microsecond=0) - last_date_data
            hours_difference = time_difference.total_seconds() / 3600 # Get the number of hours
            missing_data = self.fetch_latest_data(limit = int(hours_difference) + 1)
            complete_data = pd.concat([data, missing_data])
            complete_data.index = complete_data.index.set_levels(pd.to_datetime(complete_data.index.levels[0]), level=0)
            complete_data.to_csv(self.market_data_filename)
            print('Market data updated successfully')
        else:
            print('No missing data')
            data.to_csv(self.market_data_filename)
            print('Market data updated successfully')
    

    ############ Main Methods ############
    def upload_complete_market_data(self, data_size = 2200):
        start_time = (dt.datetime.now() - dt.timedelta(hours= self.max_rows_market_data)).date()
        end_time = dt.datetime.now().date()
        timeframes = ['1w', '1d', '4h', '1h', '30m','15m', '5m', '1m']
        index = 3 #It is better to choose the highest frequency for the backtest to be able to downsample
        interval = timeframes[index]
        self.data_instance = Data(self.symbols_to_trade, interval, start_time, end_time, exchange = 'kraken')
        data = self.data_instance.df
        
        self.complete_missing_data(data)

    def fetch_latest_data(self, limit=2):
        """Fetch latest OHLCV data for multiple symbols and stack them into a single DataFrame."""
        
        formatted_symbols = self.format_symbols(self.symbols_to_trade)
        
        def fetch_symbol_data(symbol, formatted_symbol):
            """Fetch data for a single symbol and return a DataFrame."""
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, self.timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df['coin'] = formatted_symbol
                return df
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
                try:
                    # Retry fetching data
                    ohlcv = self.exchange.fetch_ohlcv(formatted_symbol, self.timeframe, limit=limit)
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    df['coin'] = formatted_symbol
                    return df
                except Exception as e:
                    print(f"Error fetching data for {symbol} on retry: {e}")
                    return pd.DataFrame()

        # Use ThreadPoolExecutor for parallel requests
        with ThreadPoolExecutor(max_workers=16) as executor:  # Adjust workers based on CPU
            results = list(executor.map(fetch_symbol_data, self.symbols_to_trade, formatted_symbols))

        # Concatenate all DataFrames and set multi-level index
        data_frames = [df for df in results if not df.empty]
        if data_frames:
            stacked_df = pd.concat(data_frames)
            stacked_df.set_index('coin', append=True, inplace=True)
            stacked_df = stacked_df[~stacked_df.index.duplicated()]  # Remove duplicates
            df = self.data_instance.prepare_data(stacked_df.unstack())
            df.reset_index(level = 1, inplace = True)
            df['coin'] = df['coin'].str.replace('/USD', 'USDT', regex=False).replace('USD', 'USDT', regex=False)
            df.set_index('coin', append = True, inplace = True)
            return df
        else:
            return pd.DataFrame()  # Return an empty DataFrame if no data
            
        # Append new data to CSV and maintain max length (asynchronous)
    def append_to_csv_with_limit(self, filename, latest, use_limit = True, last_row = True):
        """_summary_

        Args:
            data (_type_): _description_
            filename (_type_): _description_
            max_rows (int, optional): _description_. Defaults to 2202. Should be account for the max number of rows needed for any of the processes
        """
        file_exists = os.path.isfile(filename)
        
        if len(latest) == 0:
            print('No data to append', latest)
            return
        latest_data = latest.loc[latest.index.get_level_values(0).unique()[-1]]
        last_index = [latest.index.get_level_values(0).unique()[-1]] * len(latest_data)
        latest_data.index = pd.MultiIndex.from_tuples(zip(last_index, latest_data.index), names = ['date', ''])
        
        if file_exists and os.path.getsize(filename) > 0:
            existing_df = pd.read_csv(filename, index_col=[0, 1], parse_dates=['date'])
            print(f'Last date of the existing data inside the file: {existing_df.index.get_level_values(0).unique()[-1]}')
            print(f'Last date of the latest data inside the file: {latest.index.get_level_values(0).unique()[-1]}')

            if existing_df.index.get_level_values(0).unique()[-1] == latest.index.get_level_values(0).unique()[-1]:
                return
            
            if last_row:
                combined_df = pd.concat([existing_df, latest_data])
            else:
                combined_df = pd.concat([existing_df, latest])

            if len(combined_df) > self.max_rows_market_data and use_limit:
                combined_df = combined_df.unstack().iloc[-self.max_rows_market_data:].stack(future_stack=True)
                print('Sliced Combined Dataframe Successfully')
            combined_df.to_csv(filename)
        else:
            print('File does not exist or is empty. Adding data to it.')
            if last_row:
                latest_data.to_csv(filename, mode='w', header=True)
            else:
                latest.to_csv(filename, mode = 'w', header=True)
            
    #Getting the data from csv
    def load_data_from_csv(self):
        filename = self.market_data_filename
        if os.path.isfile(filename):
            try:
                data = pd.read_csv(filename, index_col=[0, 1], parse_dates=['date'])
                if len(data.unstack()) >= self.train_size + self.test_size:
                    print(f'Returning data. Its size: {len(data.unstack())}')
                    return data
                else:
                    print(f'Data not large enough. Its size: {len(data.unstack())}')
                    return
            except Exception as e:
                print(f'File does not exist or is empty: {e}')
        else:
            print(f'The file is empty or does not exist')
        
    
    
    def perform_portfolio_rm(self):
        
        if os.path.isfile(self.strategy_data_filename) and os.path.getsize(self.strategy_data_filename) > 0:
            try:
                current_strategy_returns_df = pd.read_csv(
                    self.strategy_data_filename,
                    index_col=[0, 1],
                    parse_dates=['date']
                )
            except pd.errors.EmptyDataError:
                print("The file is empty or has no valid data.")
                return False
        else:
            print(f"File {self.strategy_data_filename} does not exist or is empty.")
            return False


        if current_strategy_returns_df.empty or len(current_strategy_returns_df) < self.train_size + self.test_size:
            return False
        portfolio_returns = current_strategy_returns_df.dot(self.best_weights)
        portfolio_returns_series = pd.Series(portfolio_returns)
        
        
        # ######## Plotting the portfolio returns each loop ########
        # plt.ion()  # Turn on interactive mode
        # fig, ax = plt.subplots()
        # portfolio_cumulative_returns = portfolio_returns.cumsum().apply(np.exp)

        # # Update the plot data here
        # ax.clear()  # Clear the previous plot
        # portfolio_cumulative_returns.plot(ax=ax)  # Re-plot the data
        # plt.draw()  # Update the plot with new data
        # plt.pause(0.1)  # Pause for a short time to allow for updates
        # ############################################################
        
        

        portfolio_rm_instance = Portfolio_RM(portfolio_returns_series)

        drawdown_limit, in_drawdown = portfolio_rm_instance.drawdown_limit(self.drawdown_threshold)

        if in_drawdown.iloc[-1]:
            #Liquidate the portfolio
            print(f'Liquidating the portfolio because in_drawdown in {in_drawdown.iloc[-1]}')
            symbols_to_liquidate = self.symbols_in_current_balance()
            symbols_to_liquidate = [s.replace('USDT', '') for s in symbols_to_liquidate]
            self.liquidate(symbols_to_liquidate)
            return True
        else :
            print(f'Portfolio is not in drawdown because in drawdown is {in_drawdown.iloc[-1]}')
            return False
    
    def run_wfo_and_get_results_returns(self):
        """_summary_
        Takes the strategy map, runs the WFO for each strategy and returns the results of the strategy returns after the WFO.
        It also adds the df of the strategy returns to a csv file

        Args:
            strategy_map (_type_): _description_

        Returns:
            results_strategy_returns (_type_): _description_ the results of the strategy returns after the WFO
        """
        #Run the WFO for each strategy (but the cash strategy)
        for key, value in self.strategy_map.items():
            if key != 'cash_strat':
                value.test()
                
        #Make a new dictionary that contains the results strategy returns of the WFO
        results_strategy_returns = {}
        for key, value in self.strategy_map.items():
            if key != 'cash_strat':
                results_strategy_returns[key] = value.results.strategy
            elif key == 'cash_strat':
                results_strategy_returns[key] = value.strategy
        
        return results_strategy_returns
            
    def perform_portfolio_management(self):
        """_summary_

        Args:
            strategy_map (_type_): _description_
            low_corr_threshold (int, optional): _description_. Defaults to 1.
        """
        if self.counter % self.portfolio_optimization_frequency == 0:
            print('Already have the results strategy returns from the portfolio optimization, Skipping the WFO Process...')
            results_strategy_returns = self.results_strategy_returns_
        else:
            results_strategy_returns = self.run_wfo_and_get_results_returns()

        portfolio_management = Portfolio_Management(results_strategy_returns)

        keys_for_selected_strategy = portfolio_management.filter_by_correlation(low_corr_threshold= self.low_corr_thresh).columns

        self.selected_strategy = {key: value for key, value in self.strategy_map.items() if key in keys_for_selected_strategy}
        
        self.live_selected_strategy = {key: value for key, value in self.live_strategy_map.items() if key in keys_for_selected_strategy}
        
    def perform_optimization(self):
        """_summary_

        Args:
            strategy_map (_type_): _description_
        """

        #Run the optimization to get the strategy parameters
        for key, value in self.strategy_map.items():
            if key != 'cash_strat':
                value.optimize()

        #Storing the best_params for each strategy in a separate dictionary
        self.best_params = {key: value.best_params for key, value in self.strategy_map.items() if key != 'cash_strat'}
        
    def perform_portfolio_optimization(self):
        """_summary_

        Args:
            strategy_returns_df (_type_): _description_
            train_size (int, optional): _description_. Defaults to 1000.
            test_size (int, optional): _description_. Defaults to 1000.
            step_size (int, optional): _description_. Defaults to 1000.
        """
        results_strategy_returns = self.run_wfo_and_get_results_returns()
        
        self.results_strategy_returns_ = results_strategy_returns
        
        #Get portfolio optimization instance
        portfolio_optimization_instance = Portfolio_Optimization(log_rets = results_strategy_returns, train_size = self.train_size, test_size = self.test_size, step_size = self.step_size, objective = 'multiple')

        #Run the 
        results_strategy_returns_df = pd.concat(results_strategy_returns, axis = 1).fillna(0)
        train_data = results_strategy_returns_df.iloc[-self.train_size:]
        self.best_weights = portfolio_optimization_instance.optimize_weights_minimize(train_data)
    
    def run_strategy(self):
        #Get the current_total_balance
        current_total_balance = self.get_portfolio_value()
        
        # ###################################
        # live_strat_1_instance = Last_Days_Low(self.data, objective='multiple', train_size=500, test_size=500, step_size=500, live = True)
        # live_strat_2_instance = Sprtrnd_Breakout(self.data, objective='multiple', train_size=500, test_size=500, step_size=500, live = True)
        # self.live_selected_strategy = {
        #     'cash_strat': self.cash_df,
        #     'strat_1': live_strat_1_instance,
        #     'strat_2': live_strat_2_instance,
        # }
        # self.best_weights = [0.0, 0.5, 0.5]
        # self.best_params = {'strat_1': {'std_window': np.int64(19), 'mean_window': np.int64(6), 'ema_window': np.int64(85), 'hourly_lookback': np.int64(3), 'daily_lookback': np.int64(3), '_min_pos': 0.6949941493453458, '_max_pos': 1.0909079937846315, 'sl_ind_length': np.int64(20), 'sl_ind_mult': 3.066997884824298, 'tp_mult': np.int64(5), 'ptp_mult': 1.9611720243493493, 'ptp_exit_percent': 0.8600804638103364},
        #     'strat_2': {'std_window': np.int64(22), 'mean_window': np.int64(22), 'ema_window': np.int64(61), 'str_length': np.int64(17), 'str_mult': np.int64(3), '_min_pos': 0.28719515606534246, '_max_pos': 1.457568143083656, 'sl_ind_length': np.int64(43), 'sl_ind_mult': 3.7477828452419297, 'tp_mult': np.int64(3), 'ptp_mult': 1.256068322761324, 'ptp_exit_percent': 0.1363902305845882}}
        # ###################################
        
        print(f"Best Weights: {self.best_weights}")
        print(f"Current Total Balance: {current_total_balance}")
        print(f"Live Selected Strategy: {self.live_selected_strategy}")
        print(f'Selected Strategy: {self.selected_strategy}')
        print(f'Live Strategy Map: {self.live_strategy_map}')
        #Store the max allocation for each strategy in a dictionary
        max_allocation_map = {
            key: self.best_weights[i] * current_total_balance / strategy.max_universe
            for i, (key, strategy) in enumerate(self.live_selected_strategy.items())
            if i < len(self.best_weights) and self.best_weights[i] > 0 and key != 'cash_strat'
        }

        print(f'Max_allocation_map: {max_allocation_map}')
        #Rebuild the strategy map, with the updated max_allocation for each strategy
        for key, value in self.live_selected_strategy.items():
            if key != 'cash_strat':
                value.max_dollar_allocation = max_allocation_map.get(key, 0)
                print(f"Max Dollar Allocation for {key}: {value.max_dollar_allocation}")
            
        print('Fetching latest market data...')
        latest = self.fetch_latest_data()
        print('Fetching Done. Appending it to market data...')
        self.append_to_csv_with_limit(self.market_data_filename, latest)
        print('Appending done. Loading data...')
        data = self.load_data_from_csv()
        print(f'Loading done. Data head: {data.head()}')
        
        
        #Run each strategy on enough data points and get the total portfolio value
        data_to_run_strategy = data.unstack().iloc[-self.length_of_data_to_run_strategy:].stack(future_stack = True)
        print(f'Data to run the strategy on: {data_to_run_strategy}')
        
        current_strategy_results = {
            key: value.trading_strategy(data_to_run_strategy, self.best_params[key])
            for key, value in self.live_selected_strategy.items()
            if key != 'cash_strat'
        }

        for key, value in current_strategy_results.items():
            if 'strategy' in value.columns:
                print(f'Strategy Column in {key}: {value.head()}')
            else:
                print(f'Strategy not in columns. All other columns for {key}: {value.head()}')
                
        
        current_strategy_returns = {
            key: value['strategy']
            for key, value in current_strategy_results.items()
        }
        
        for key, value in current_strategy_returns.items():
            print(f"Strategy returns for {key}: {value}")
        
        #Append current strategy results to the csv file for future analysis
        current_strategy_returns_df = pd.concat(current_strategy_returns, axis=1).fillna(0)
        cash_strategy = self.cash_df['strategy'].reindex(current_strategy_returns_df.index).dropna()
        current_allocation_results_df = pd.concat([current_strategy_returns_df, cash_strategy], axis=1).fillna(0)
        
        print(f'Current Strategy returns df: {current_strategy_returns_df}')
        print('Appending to Strategy returns data...')
        self.append_to_csv_with_limit(self.strategy_data_filename, current_strategy_returns_df, use_limit = False, last_row = False)
        print(f'Appending Done.')
        
        
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
    
        
        # Extract current universes from selected_strategy
        print('Getting Universe')
        current_universes = [
            set(value.current_universe)  # Convert each universe to a set for comparison
            for key, value in self.live_selected_strategy.items()
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


        symbols_in_current_balance = self.symbols_in_current_balance()
        print(f'Symbols in Current balance: {symbols_in_current_balance}')
        
        # Ensure symbols_in_current_balance is not None
        if symbols_in_current_balance:
            symbols_not_in_universe = [
                symbol.replace('USDT', '').replace('USD', '') for symbol in symbols_in_current_balance
                if symbol not in flattened_universe
            ]
            print(f"Liquidating {symbols_not_in_universe}...")
            self.liquidate(symbols_not_in_universe)
            print("Liquidation complete.")
        else:
            print("symbols_in_current_balance is None or empty.")

        print(f'Current_universe: {flattened_universe}')
        for coin in flattened_universe:
            formatted_coin = coin.replace('USDT', '').replace('USD', '')
            coin_for_order = coin.replace('USDT', '/USD')
            coin_balance = self.get_coin_balance(formatted_coin)
            current_coin_allocation = current_allocation[coin]
            
            if coin_balance is None:
                coin_balance = 0
            
            print(f'Current coin allocation: {current_coin_allocation}')
            print(f'Coin balance: {coin_balance}')
            to_add = round(current_coin_allocation - coin_balance, 7)
            
            
            if to_add > 0 and to_add < self.get_usd_left():
                print(f"Adding {to_add} {formatted_coin} to the portfolio...")
                self.buy(to_add, coin_for_order)
            elif to_add < 0 and coin_balance >= abs(to_add):
                print(f"Selling {-to_add} {formatted_coin} from the portfolio...")
                self.sell(-to_add, coin_for_order)
            else:
                print(f"Nothing to add because {to_add} in coin's currency is almost $0.0")
            
        
    def main_loop(self):
        # THE MAIN LOOP
        while True:

            if self.counter % self.strategy_optimization_frequency == 0:
                print('Performing optimization')
                self.perform_optimization()

            if self.counter % self.portfolio_optimization_frequency == 0:
                print('Performing portfolio optimization')
                self.perform_portfolio_optimization()

            if self.counter % self.portfolio_management_frequency == 0:
                print('Performing portfolio management')
                self.perform_portfolio_management()
            
            print('Adding to counter')
            self.counter += 1
            
            data = pd.read_csv(self.market_data_filename, index_col=[0, 1], parse_dates=['date'])
            self.complete_missing_data(data)
            print('Updating Data Before Portfolio RM')

            if self.perform_portfolio_rm():
                print('Performed portfolio risk management, portfolio is in drawdown')
                now = dt.datetime.now()  # Skip running the strategy, go straight to time update
                print('Current time: ', now)
                next_hour = (now + dt.timedelta(hours=1)).replace(minute = 0, second=0, microsecond=0)
                print('Next hour: ', next_hour)
                sleep_duration = (next_hour - now).total_seconds()
                print('Sleep duration: ', sleep_duration)
                time.sleep(sleep_duration)
                continue  # Skip the strategy execution and restart the loop
            else:
                print('Portfolio is not in drawdown')
            

            #Perform the strategy after each hour
            now = dt.datetime.now()
            print('Current time: ', now)
            next_hour = (now + dt.timedelta(hours=1)).replace(minute = 0, second=0, microsecond=0)
            print('Next hour: ', next_hour)
            sleep_duration = (next_hour - now).total_seconds()
            print('Sleep duration: ', sleep_duration)
            time.sleep(sleep_duration)
            
            print('Running strategy')
            self.run_strategy()