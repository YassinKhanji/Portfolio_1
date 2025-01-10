import pandas as pd
import numpy as np



class Portfolio_RM():
    def __init__(self, returns):
        """
        returns: (pd.DataFrame, pd.Series) Returns of the overall the portfolio
        """
        self.returns = returns
    
    def drawdown_limit(self, threshold):
        # Step 1: Calculate the cumulative returns
        creturns = self.returns.cumsum().apply(np.exp)
        
        # Step 2: Calculate cumulative maximum
        cumulative_max = creturns.cummax()

        # Step 3: Calculate drawdown
        drawdown = (creturns - cumulative_max) / cumulative_max

        # Step 4: Calculate the drawdown limit
        drawdown_limit = np.where(drawdown < threshold, 0, self.returns)
        
        return pd.Series(drawdown_limit)