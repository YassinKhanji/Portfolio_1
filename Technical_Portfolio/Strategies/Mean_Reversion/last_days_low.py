import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os
from typing import List
import datetime as dt
from skopt.space import Integer, Real, Categorical



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Data_Management')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Universe_Selection')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Signal_Generation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','Risk_Management')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'Validation')))

from data import Data
from fetch_symbols import get_symbols
from calculations import Calculations, Metrics
from coarse import Coarse_1 as Coarse
from fine import Fine_1 as Fine
from entry_signal import Trend_Following, Mean_Reversion
from tail_risk import Stop_Loss, Take_Profit
from position import Position
from manage_trade import Manage_Trade
from testing import WFO
from costs import Costs
from stress_test import Stress_Test

class Last_Days_Low():
    def __init__(self, df, optimize_fn="gp", 
                            objective='sharpe', 
                            opt_freq='custom', 
                            num_simulations = 1000,
                            confidence_level = 0.95, 
                            blocks = 20,
                            max_universe = 4,
                            live = False,
                            train_size = 2200,
                            test_size = 2200,
                            step_size = 2200):
        self.df = df.copy()
        self.max_universe = max_universe
        self.optimize_fn = optimize_fn
        self.objective = objective
        self.opt_freq = opt_freq
        self.live = live
        self.param_space = {
        'std_window': Integer(5, 30),
        'mean_window': Integer(5, 30),
        'ema_window': Integer(5, 100),
        'hourly_lookback': Integer(1, 5),
        'daily_lookback': Integer(1, 5),
        '_min_pos': Real(0, 0.75),
        '_max_pos': Real(0, 1),
        'sl_ind_length': Integer(5, 50),
        'sl_ind_mult': Real(0.5, 5),
        'tp_mult': Integer(2, 7),
        'ptp_mult': Real(1, 2),
        'ptp_exit_percent': Real(0.1, 1)
        }
        self.all_frequency = ['1W', '1D', '4h','1h', '30min','15min', '5min', '1min'] #All possible frequencies for the resampling 
        self.current_universe = set()
        
        self.performance = -np.inf
        self.results = None
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        
        self.best_params = None
        self.cum_strategy = None
        
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self.blocks = blocks
        self.overall_score = 0.0
        self.metrics_df = None
        self.sims = None
        
        

        

    
    ######## Helper Methods ########
    def update_universe(self, df: pd.DataFrame, max_positions: int = 4, low_freq = '1d') -> pd.Series:
        """
        Updates a DataFrame to track a dynamic universe of coins.
        Should include the dataframe with the lower frequency data. (daily, weekly, etc.)
        Assumes a stacked dataframe
        """
        threshold = 0.1
        current_universe = set()
        df['in_universe'] = False
        df['position'] = np.where(df['position'] < threshold, 0, df['position']) #Since when optimizing, min_pos can never be 0, thus we put a threshold of 0.1 to indicate a non position    


        for time_index in df.index.get_level_values(0).unique():
            # Remove coins that are no longer in the universe *for this time index*
            coins_to_remove = []
            
            
            for coin in current_universe:
                if (time_index, coin) in df.index and df.loc[(time_index, coin), 'position'] == 0: 
                    coins_to_remove.append(coin)
                    df.loc[(time_index, coin), 'in_universe'] = False
            current_universe.difference_update(coins_to_remove) #use difference_update for set manipulation


            current_coins = df.loc[time_index].index
            available_coins = set(current_coins) - current_universe

            if len(current_universe) < max_positions and available_coins:
                temp_df = df.loc[(time_index, list(available_coins)), :].copy()

                # The shift was the main source of the bug. It was shifting across coins,
                # which is incorrect. We should not shift at all in this context.
                # The intention was likely to use the *previous* time slice data.
                # This is handled later.

                filter_condition = (
                    (temp_df['above_ema']) &
                    (temp_df['volume_rank'] < 50) &
                    (temp_df['std_rank'] < 10) &
                    (temp_df['entry_signal'] == 1)
                )

                potential_coins_df = temp_df[filter_condition]

                if not potential_coins_df.empty:
                    potential_coins_df = potential_coins_df.sort_values(by='std_rank')
                    potential_coins = set(potential_coins_df.index.get_level_values(1))
                    missing_positions = max_positions - len(current_universe)
                    to_be_added: List[str] = list(potential_coins)[:missing_positions]
                    current_universe.update(to_be_added)

            df.loc[(time_index, list(current_universe)), 'in_universe'] = True
        
        df = df.unstack()
        df['in_universe'] = df['in_universe'].shift(periods = 1, freq = low_freq)
        df = df.stack(future_stack= True)
        return df['in_universe'], current_universe

    def trading_strategy(self,
        data,
        params = None,
        ###### To Optimize ######
        std_window = 20,
        mean_window = 20,
        ema_window = 20,
        hourly_lookback = 1,
        daily_lookback= 1,
        _min_pos = 0, #Has to be >= 0
        _max_pos = 1, #Has to be > 0
        sl_ind_length = 14,
        sl_ind_mult = 3,
        tp_mult = 2,
        ptp_mult = 1,
        ptp_exit_percent = 0.5,
        ###### Constants ######
        low_freq_index = 1, #The index of the lowest frequency for the resampling
        high_freq_index = 3, #The index of the highest frequency for the resampling
        max_perc_risk = 0.01,
        max_dollar_allocation = 10000,
        sl_type = 'atr',
        tp_type = 'rr',
        sl_signal_only = True,
        tp_signal_only = True,
        ptp_signal_only = True,
        tp_ind_length = 0,
        fixed_sl = True,
        fixed_tp = True,
        maker = 0.25,
        taker = 0.40
        ):
    
        if params is not None:
            if isinstance(params, list):
                std_window = params[0]
                mean_window = params[1]
                ema_window = params[2]
                hourly_lookback = params[3]
                daily_lookback = params[4]
                _min_pos = params[5]
                _max_pos = params[6]
                sl_ind_length = params[7]
                sl_ind_mult = params[8]
                tp_mult = params[9]
                ptp_mult = params[10]
                ptp_exit_percent = params[11]
            if isinstance(params, dict):
                std_window = params['std_window']
                mean_window = params['mean_window']
                ema_window = params['ema_window']
                hourly_lookback = params['str_length']
                daily_lookback = params['str_mult']
                _min_pos = params['_min_pos']
                _max_pos = params['_max_pos']
                sl_ind_length = params['sl_ind_length']
                sl_ind_mult = params['sl_ind_mult']
                tp_mult = params['tp_mult']
                ptp_mult = params['ptp_mult']
                ptp_exit_percent = params['ptp_exit_percent']
            


        if high_freq_index > low_freq_index:
            low_freq = self.all_frequency[low_freq_index] #The lowest frequency for the resampling
            high_freq = self.all_frequency[high_freq_index], #The highest frequency for the resampling
                #Generally not going to be used since we are not calling the data inside the function
        
        #########################
        
        cal = Calculations()
        mr = Mean_Reversion()
        #Generate a signal
        print(f'length of data: {len(data)}')
        _df = mr.last_days_low(data.copy(), hourly_lookback, daily_lookback)
        print('Signal Generated')

        pos = Position(_df, _min_pos, _max_pos, self.live)
        _df = pos.initialize_position()
        print('Position Initialized')
        sl = Stop_Loss(_df, sl_type, sl_ind_length, sl_ind_mult, sl_signal_only)
        _df = sl.apply_stop_loss(fixed_sl, plot = False)
        print('Stop Loss Applied')
        tp = Take_Profit(_df, tp_type, tp_mult, tp_signal_only)
        _df = tp.apply_take_profit(fixed_tp, plot = False)
        print('Take Profit Applied')
        ptp = Take_Profit(_df, tp_type, ptp_mult, ptp_signal_only, exit_percent = ptp_exit_percent)
        _df = ptp.apply_take_profit(fixed_tp, plot = False)
        print('Partial Take Profit Applied')

        _df = cal.merge_cols(_df, common = 'exit_signal', use_clip = True)
        _df = pos.calculate_position(_df)
        print('Position Calculated')

        mt = Manage_Trade(_df)
        _df = mt.erw_actual_allocation(max_perc_risk, max_dollar_allocation)
        print('Manage Trade Applied')
        
        _df = cal.update_all(_df)
        print('All Updated')

        #########################
        
        #Calculate transaction costs on strategy returns
        costs = Costs(_df, maker = maker, taker = taker)
        df = costs.apply_fees() #Applies fees on strategy returns, appears on cumulative returns when applied 
        print('Costs Applied')

        #########################
        
        #Downsample the data
        df = cal.downsample(df, low_freq)
        print('Data Downsampled')

        #Perform coarse analysis and filtering
        coarse = Coarse()
        df = coarse.volume_flag(df, max_dollar_allocation)
        df = coarse.sort_by_volume(df)
        df = coarse.sort_by_std(df, std_window, mean_window)
        print('Coarse Analysis Done')
        fine = Fine()
        df = fine.above_ema(df, ema_window)
        print('Fine Analysis Done')

        #apply update_univers
        df['in_universe'], self.current_universe = self.update_universe(df, max_positions = self.max_universe)
        print('Universe Updated')

        df.dropna(inplace = True)

        df = df[df['in_universe']]
        print('In Universe')
        
        return df



    ######## Main Methods ########    
    def optimize(self, test = False) -> pd.DataFrame:
        """
        Optimize the strategy using the Gaussian Process optimizer

        Returns:
            optimized: The optimized results for the test period
        
        !!! Make Sure to Run the test method first to get the best parameters for the optimization !!!

        """
        wfo = WFO(self.df,
                self.trading_strategy,
                self.param_space)
        
        test_size = self.test_size if test else 0
        self.train_data = self.df.iloc[-self.train_size + test_size:]
        self.best_params = wfo.optimize_parameters_gp(self.train_data, self.param_space)

        if test:
            self.test_data = self.df.iloc[-self.test_size:]
            optimized_df = wfo.test_strategy(self.test_data, self.best_params)
            return optimized_df[1]
    
    def test(self) -> None:
        """
        Test the strategy using the best parameters from the optimization

        Returns:
            results: The results of the test

        """
        wfo = WFO(self.df, 
                    self.trading_strategy, 
                    self.param_space, 
                    train_size=self.train_size, 
                    test_size=self.test_size, 
                    step_size=self.step_size, 
                    optimize_fn=self.optimize_fn, 
                    objective=self.objective, 
                    opt_freq=self.opt_freq)

        self.performance, self.results = wfo.walk_forward_optimization()
        self.results['cstrategy'] = self.cum_strategy = (self.results['strategy'] * (1/self.max_universe)).cumsum().apply(np.exp)

    def stress_test(self) -> None:
        """
        Perform a stress test on the strategy, uses block bootstrap to simulate different paths        
        """
        strategy = self.results['strategy']
        stress_test = Stress_Test(strategy, self.num_simulations, self.confidence_level, len(self.results))
        self.sims = stress_test.block_bootstrap(self.blocks)
        self.metrics_df = stress_test.metrics_df_fnct(self.sims)
        self.overall_score = stress_test.score_strategy(self.metrics_df)
    
        