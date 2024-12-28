import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import datetime as dt


class Data:
    def __init__(self, symbols, interval, start_time, end_time):
        self.symbols = symbols
        self.interval = interval
        self.start_time = start_time
        self.end_time = end_time
        self.available_symbols = self.binance_symbols()

    def binance_symbols(self):
        """Fetch available symbols from Binance API."""
        response = requests.get("https://api.binance.com/api/v3/exchangeInfo")
        exchange_info = response.json()
        valid_symbols = {s['symbol'] for s in exchange_info['symbols']}
        return [s for s in self.symbols if s in valid_symbols]

    def fetch_symbol_data(self, symbol, date_list, url, limit):
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
            if isinstance(data, list):
                all_data.extend(data)
        return symbol, all_data

    def get_binance_klines(self, limit=1000):
        """Fetch historical kline data for all symbols in parallel."""
        url = "https://api.binance.com/api/v3/klines"
        date_list = pd.date_range(start=self.start_time, end=self.end_time, freq='D').tolist()

        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(
                lambda symbol: self.fetch_symbol_data(symbol, date_list, url, limit),
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
        df = df.copy()
        for coin in df.columns.levels[1]:
            df['returns', coin] = df['close', coin].pct_change()
            df['log_return', coin] = np.log1p(df['returns', coin])
            df["creturns", coin] = df["log_return", coin].cumsum().apply(np.exp)
            df['price', coin] = df['close', coin]
            df['volume_in_dollars', coin] = df['close', coin] * df['volume', coin]

        df = df.stack(level=1)
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


# Example usage
symbols = ['BTCUSDT', 'ETHUSDT']
interval = '1h'
start_time = dt.datetime(2020, 1, 1)
end_time = dt.datetime(2020, 3, 1)

data_instance = Data(symbols, interval, start_time, end_time)
df = data_instance.get_data()
print(df)