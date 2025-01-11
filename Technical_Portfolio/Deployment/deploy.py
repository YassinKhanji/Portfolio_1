class deploy():
    def __init__(self):
        pass
    
    @unsync
    def fetch_latest_data(self, symbol, timeframe):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=3)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            latest_data = df[:-1]
            return latest_data
        except Exception as e:
            print(f"Error fetching latest data: {e}")
            return None
        
    