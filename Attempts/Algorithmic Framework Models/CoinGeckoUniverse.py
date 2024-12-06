# region imports
from AlgorithmImports import *
import pandas as pd
# endregion

from typing import List

class CoinGeckoUniverse:
    def __init__(self, algorithm, categories):
        self.algorithm = algorithm
        self.categories = categories

    def fetch_symbols(self):
        url = f"https://raw.githubusercontent.com/yaquants/symbols/main/symbols_2024-09-13_00-15-39.csv" 
        # Download the CSV file content
        data = self.algorithm.download(url)
        
        # Extract symbols (assuming there's a column named 'Symbol')
        symbols = data.splitlines()

        # Log the symbols
        for symbol in symbols:
            self.algorithm.debug(f"Downloaded symbol: {symbol}")
        
        return symbols
