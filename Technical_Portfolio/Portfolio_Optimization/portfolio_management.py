import requests
import json
import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt


class Portfolio_Management():
    def __init__(self, strategy_rets_map):
        self.strategy_rets_map = strategy_rets_map   
        
    def get_filtered_strategies(self, selected, strategy_rets_map):
        """Filters the strategies based on the selected strategies."""
        filtered_strategies = {strategy_name : strategy for strategy_name, strategy in strategy_rets_map.items() if strategy_name in selected}
        return pd.concat(filtered_strategies, axis = 1)
    
    def filter_by_correlation(self, low_corr_threshold = 0.9):
        """
        Filter the strategies by correlation
        """
        #Concatenate the map, to get one dataframe
        all_strategy = pd.concat(self.strategy_rets_map, axis = 1)
    
        #Calculate the correlation matrix  
        corr_matrix = all_strategy.corr()

        #Perform the correlation analysis:
        selected_strategies = [corr_matrix.columns[1]]

        for i in range(1, len(corr_matrix.columns)):
            candidate_strategy = corr_matrix.columns[i]
            correlations_with_selected = corr_matrix.loc[selected_strategies, candidate_strategy]
            if all(corr < low_corr_threshold for corr in correlations_with_selected):
                selected_strategies.append(candidate_strategy)
        
        selected_strategies.append(corr_matrix.columns[0])
        
        
        #Get a new list for the selected strategies
        filtered_strategies = self.get_filtered_strategies(selected_strategies, self.strategy_rets_map)
        
        return filtered_strategies