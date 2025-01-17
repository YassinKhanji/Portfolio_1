import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import os
from functools import reduce
from fetch_symbols import get_symbols
import ccxt
import random


class Data:
    def __init__(self, symbols, interval = '1h', start_time = dt.datetime(2020, 1, 1), end_time = dt.datetime(2020, 1, 2), get_data = True,
                 exchange = 'binance'):
        self.symbols = symbols
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        if exchange == 'binance':
            if not self.symbols[0].endswith('T'):
                self.symbols = [s + 'T' for s in self.symbols]
            self.available_symbols = self.binance_symbols()
        elif exchange == 'kraken':
            self.available_symbols = self.kraken_symbols()
            
        if get_data:
            self.df = self.get_data()

    def binance_symbols(self):
        """Fetch available symbols from Binance API."""
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
        exchange_info = response.json()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols']}
        return [s for s in self.symbols if s in valid_symbols]
    
    def kraken_symbols(self):
        """Fetch available symbols from Kraken API."""
        exchange = ccxt.kraken()
        markets = exchange.load_markets()
        valid_symbols_ = {market['symbol'] for market in markets.values()}
        valid_symbols_ = [s.replace("/USD", "USD") for s in valid_symbols_ if s.endswith('USD')]
        valid_symbols_.sort()
        return [s for s in self.symbols if s in valid_symbols_]

    def fetch_symbol_binance_data(self, symbol, date_list, url, limit):
        """Fetch kline data for a single symbol."""
        all_data = []
        for i in range(len(date_list) - 1):
            params = {
                'symbol': symbol,
                'interval': self.interval,
                'startTime': int(date_list[i].timestamp() * 1000),
                'endTime': int((date_list[i + 1] - dt.timedelta(seconds=1)).timestamp() * 1000),
                'limit': limit,
            }
            response = requests.get(url, params=params)
            data = response.json()
            if isinstance(data, list) and data:
                all_data.extend(data)
        return symbol, all_data

    def get_binance_klines(self, limit=1000):
        """Fetch historical kline data for all symbols in parallel."""
        url = "https://api.binance.com/api/v3/klines"
        date_list = pd.date_range(start=self.start_time, end=self.end_time, freq='D').tolist()
        
        if not self.available_symbols[0].endswith('T'):
            self.available_symbols = [s + 'T' for s in self.available_symbols]

        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(
                lambda symbol: self.fetch_symbol_binance_data(symbol, date_list, url, limit),
                self.available_symbols,
            )

        
        # Process and combine results
        data_frames = {}
        for symbol, data in results:
            if not data:
                continue
            df = pd.DataFrame(data)
            df = df.iloc[:, 0:6]
            df.columns = ['Open Time', 'open', 'high', 'low', 'close', 'volume']
            df.index = pd.to_datetime(df['Open Time'], unit='ms')
            df.drop('Open Time', axis=1, inplace=True)
            data_frames[symbol] = df

        if not data_frames:
            return None

        combined_df = pd.concat(data_frames, axis=1)
        combined_df = combined_df.swaplevel(axis=1).sort_index(axis=1)
        combined_df = combined_df.apply(pd.to_numeric, errors='coerce')

        return combined_df

    def prepare_data(self, df):
        """Prepare data for analysis."""
        _df = df.copy()
        for coin in df.columns.levels[1]:
            _df['returns', coin] = _df['close', coin].pct_change(fill_method=None)
            _df['log_return', coin] = np.log(_df['returns', coin] + 1)
            _df["creturns", coin] = _df["log_return", coin].cumsum().apply(np.exp)
            _df['price', coin] = _df['close', coin]
            _df['volume_in_dollars', coin] = _df['close', coin] * _df['volume', coin]

        df = _df.stack(future_stack=True).copy()
        df.sort_index(axis=1, inplace=True)
        df.index.names = ['date', 'coin']
        df.dropna(inplace=True)

        return df

    def upload_data(self, df, filename):
        """Save data to a CSV file."""
        df.to_csv(filename)

    def get_data(self):
        """Main function to fetch, prepare, and save data."""
        df = self.get_binance_klines()
        if df is not None:
            df = self.prepare_data(df)
            self.upload_data(df, 'data.csv')
        return df
    



class CSV_Data:
    def __init__(self, folder_path, symbols):
        self.folder_path = folder_path
        self.symbols = symbols
        self.df = self.process_folder(folder_path, symbols)
        self.df = self.prepare_data()
        self.upload_data_to_csv(self.df)
        
    
    def prepare_data(self):
        df = self.df.copy()
        for coin in df.columns.levels[1]:
            df['returns', coin] = df['close', coin].pct_change()
            df['log_return', coin] = np.log(df['returns', coin])
            df["creturns", coin] = df["log_return", coin].cumsum().apply(np.exp)
            df['price', coin] = df['close', coin]
            df['volume_in_dollars', coin] = df['close', coin] * df['volume', coin]

        df = df.stack(level=1, future_stack=True)
        df.sort_index(axis=1, inplace=True)
        df.index.names = ['date', 'coin']
        df.dropna(inplace=True)

        return df
    
    def get_data(self, file_path, symbols):
        df = pd.read_csv(file_path)
        df = df.drop(columns = df.columns[-1]).reset_index()
        df.drop(columns = df.columns[0], inplace = True)
        df.drop(index = 0, inplace = True)
        df.columns = ['date', 'coin', 'open', 'high', 'low', 'close', 'volume', 'volume_in_dollars']

        if not df['coin'].iloc[0] in symbols:
            return
        # Clean the date column by stripping whitespace
        df['date'] = df['date'].str.strip()
        # Parse the date column with mixed format
        df['date'] = pd.to_datetime(df['date'], format='mixed', errors='coerce')
        
        df.set_index([df.columns[0], df.columns[1]], inplace = True)
        df = df.unstack()
        return df
    
    def process_folder(self, folder_path, symbols):
        # Get all CSV files in the folder
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        dfs = []
        
        for file in csv_files:
            file_path = os.path.join(folder_path, file)
            df = self.get_data(file_path, symbols)
            if df is not None:
                dfs.append(df)
        

        # Get the union of all indices (dates) to align the data
        all_dates = reduce(pd.Index.union, [df.index.get_level_values(0) for df in dfs])

        # Reindex all DataFrames to the same set of dates (adding NaNs where data is missing)
        dfs_aligned = [df.reindex(all_dates, level=0, fill_value=None) for df in dfs]

        # Concatenate all DataFrames
        concatenated_df = pd.concat(dfs_aligned, axis=1)
        concatenated_df = concatenated_df.sort_index(axis=1)
        concatenated_df = concatenated_df.apply(pd.to_numeric, errors='coerce', downcast='float') #Essential to perform calculations
        
        return concatenated_df

    def upload_data_to_csv(self, df):
        # Upload the data to CSV file
        df.to_csv('all_data.csv')
    
    
def get_symbols_for_bot():
    """Get the symbols for the bot to trade"""
    symbols = ['BONKUSD','ETHWUSD','OXTUSD','BLZUSD','LINKUSD','AKTUSD','RADUSD','LUNAUSD','TRACUSD','DOTUSD',
                     'OMNIUSD','XRTUSD','APUUSD','ZECUSD','QTUMUSD','AVAXUSD','PSTAKEUSD','SHIBUSD','ATHUSD','DENTUSD',
                     'PUFFERUSD','INTRUSD','KASUSD','STORJUSD','L3USD','CTSIUSD','STGUSD','RLCUSD','GMTUSD','OCEANUSD',
                     'IMXUSD','MEWUSD','ENJUSD','FXSUSD','TEERUSD','BCHUSD','COTIUSD','NEIROUSD','AGLDUSD','FTMUSD',
                     'LDOUSD','XMRUSD','MXCUSD','TIAUSD','MNTUSD','SOLUSD','POLUSD','TNSRUSD','ETHFIUSD','MASKUSD',
                     'TOKEUSD','LSKUSD','APEUSD','RENDERUSD','SUIUSD','KINTUSD','MANAUSD','CHRUSD','ENSUSD','OPUSD',
                     'STRKUSD','SANDUSD','POPCATUSD','FLRUSD','TONUSD','HONEYUSD','REZUSD','CVCUSD','CELRUSD','ETCUSD',
                     'WIFUSD','MEMEUSD','PONKEUSD','FORTHUSD','TRXUSD','DASHUSD','OGNUSD','KP3RUSD','DBRUSD','FIDAUSD',
                     'DOGEUSD','ARKMUSD','NYMUSD','MATICUSD','ATOMUSD','RARIUSD','ZRXUSD','SAGAUSD','BTCUSD','DYMUSD',
                     'KEYUSD','RUNEUSD','MINAUSD','EIGENUSD','ADAUSD','NODLUSD','NEARUSD','TAOUSD','CXTUSD','KILTUSD',
                     'ALICEUSD','LPTUSD','APTUSD','FETUSD','FLOWUSD','NOSUSD','SYNUSD','EGLDUSD','GLMRUSD','LRCUSD',
                     'PEPEUSD','LTCUSD','KSMUSD','ZROUSD','BTTUSD','GALUSD','FILUSD','SCUSD','ETHUSD','ICXUSD','ANKRUSD',
                     'TVKUSD','BLURUSD','BOBAUSD','STXUSD','PNUTUSD','MOVRUSD','ARBUSD','GOATUSD','XTZUSD','LMWRUSD',
                     'SAFEUSD','JASMYUSD','API3USD','AUDIOUSD','SEIUSD','PONDUSD','ICPUSD','RAREUSD','PHAUSD','SAMOUSD',
                     'TREMPUSD','ALTUSD','MULTIUSD','GRTUSD','WENUSD','GIGAUSD','TURBOUSD','FWOGUSD','HNTUSD','BANDUSD',
                     'MOGUSD','ZKUSD','SWELLUSD','ALGOUSD','MOODENGUSD']
    return random.sample(symbols, 50)

    



# Example usage
# symbols = ['BTCUSD', 'ETHUSD']
# symbols = get_symbols()
# # Add the symbol to each string in the list
# updated_symbols = [s + 'T' for s in symbols]
# interval = '1h'
# start_time = dt.datetime(2020, 1, 1)
# end_time = dt.datetime(2020, 1, 7)
# df = Data(updated_symbols, interval, start_time, end_time).df
# print(df)


#Use the below for uploading full data (uploaded to csv)
# symbols = get_symbols()
# binance_symbols = Data(symbols)
# folder_path = r'C:\Users\yassi\OneDrive\Documents\Trading\Algo Trading Projects\Algo Business\data\Binance Data (CSV)'
# df = CSV_Data(folder_path, symbols).df