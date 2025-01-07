import pandas as pd
import numpy as np
import datetime as dt


class Costs:
    def __init__(self, data, maker=0.1, taker=0.1):
        """
        Args:
            data (_type_): 
            maker (float, optional):  Defaults to 0.1%.
            taker (float, optional):  Defaults to 0.1%.
        """
        self.data = data
        self.maker = maker
        self.taker = taker

    def calculate_fees(self):
        pass
    
    
    def apply_fees(self):
        pass
