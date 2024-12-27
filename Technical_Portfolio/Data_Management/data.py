import numpy as np
import datetime as dt
import pandas as pd
import requests

class Data:
    """
    This class is used to get historical data from Binance API and prepare it for analysis.
    This means that each time we want to make certain analysis on the data, we can use this class to create a method to
    prepare the data and then use it in the analysis. 
    
    Attributes:
        symbols: list of strings
        interval: string (e.g. '1d', '1h', '1m')
        start_time: datetime object
        end_time: datetime object

        df: pandas DataFrame that contains the historical data for the given symbols, interval, start_time, and end_time
        """
    
    def __init__(self, symbols, interval, start_time, end_time):
        self.df = None
        self.get_binance_klines(symbols, interval, start_time, end_time)
        self.prepare_data()
        self.upload_data(self.df, 'data.csv')

    def get_binance_klines(self, symbols, interval, start_time, end_time, limit=1000):
        url = "https://api.binance.com/api/v3/klines"  # We added the endpoint to the URL so we can retrieve the klines data

        # Get a list of the dates between the two given dates
        date_list = pd.date_range(start=start_time, end=end_time, freq='D').tolist()

        data_frames = {}  # Dictionary to store dataframes for each symbol

        for symbol in symbols:
            all_df = pd.DataFrame()  # We will store all the dataframes in this dataframe

            for i in range(len(date_list) - 1):
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'startTime': int(date_list[i].timestamp() * 1000),
                    'endTime': int((date_list[i + 1] - dt.timedelta(seconds=1)).timestamp() * 1000),
                    'limit': limit
                }
                response = requests.get(url, params=params)
                data = response.json()
                if not data:
                    continue
                df = pd.DataFrame(data)
                all_df = pd.concat([all_df, df], ignore_index=True)

            if not all_df.empty:
                all_df = all_df.iloc[:, 0:6]
                all_df.columns = ['Open Time', 'open', 'high', 'low', 'close', 'volume']
                all_df.index = [dt.datetime.fromtimestamp(x / 1000.0) for x in all_df['Open Time']]
                all_df.drop('Open Time', axis=1, inplace=True)
                data_frames[symbol] = all_df

        if not data_frames:
            return None

        combined_df = pd.concat(data_frames, axis=1) # Concatenate all DataFrames and create a hierarchical index
        combined_df = combined_df.swaplevel(axis=1).sort_index(axis=1) # Swap the levels of the index and sort it
        combined_df = combined_df.apply(pd.to_numeric, errors='coerce') # Convert all columns to numeric

        self.df = combined_df

        return combined_df
    
    def prepare_data(self):
        df = self.df.copy()

        for coin in df.columns.levels[1]:
            df['returns', coin] = df['close', coin].pct_change()
            df['log_return', coin] = np.log(1 + df['returns', coin])
            df["creturns", coin] = df["log_return", coin].cumsum().apply(np.exp)
            df['price', coin] = df['close', coin] #This will refer to the prices that the strategies will be running on
                # Will eventually be modified when applying risk management
                #Essential when applying exit signals, as stop losses and take profits (mainly) won't exit at the closing
                # but would rather exit at the high or the low 
                #Usually, we would upsample close data to the lowest frequency possible to get more accurate results
                # Although in highly volatilite markets, this would provide wrong expectations. So it is more effecient to
                # actually use the exact price we exited at
            df['volume_in_dollars', coin] = df['close', coin] * df['volume', coin]

        # Sort the columns index
        df = df.stack(level=1, future_stack= True) #Stacking the index columns
        df = df.sort_index(axis=1) #Sorting by name
        df.index.names = ['date', 'coin'] #Renaming the index columns

        self.df = df
        return df
    
    def upload_data(self, df, filename):
        df.to_csv(filename)

    def symbols(self):
        symbols = []
        with open('symbols.txt', 'r') as file:
            for line in file:  # Process line by line
                words = line.split()  # Split the line into words
                for word in words:
                    symbols.append(word)
        return symbols






# Example usage
symbols = ['BTCUSDT', 'ETHUSDT']
interval = '1d'
start_time = dt.datetime(2021, 1, 1)
end_time = dt.datetime(2021, 1, 10)
data_instance = Data(symbols, interval, start_time, end_time)
df= data_instance.df
print(df)