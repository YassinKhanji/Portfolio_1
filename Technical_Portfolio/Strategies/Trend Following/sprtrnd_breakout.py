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
from position_size import Position
from manage_trade import Manage_Trade
from testing import WFO


class Sprtrnd_Breakout():
    def __init__(self, df, optimize_fn="gp", 
                            objective='sharpe', 
                            opt_freq='custom'):
        self.df = df.copy()
        self.optimize_fn = optimize_fn
        self.objective = objective
        self.opt_freq = opt_freq
        self.param_space = {
        'std_window': Integer(5, 30),
        'mean_window': Integer(5, 30),
        'ema_window': Integer(5, 100),
        'str_length': Integer(5, 50),
        'str_mult': Integer(1, 5),
        '_min_pos': Real(0, 1),
        '_max_pos': Real(1, 5),
        'sl_ind_length': Integer(5, 50),
        'sl_ind_mult': Real(0.5, 5),
        'tp_mult': Integer(2, 7),
        'ptp_mult': Real(1, 2),
        'ptp_exit_percent': Real(0.1, 1)
        }
        self.all_frequency = ['1W', '1D', '4h','1h', '30min','15min', '5min', '1min'] #All possible frequencies for the resampling 
        self.best_performance = -np.inf
        self.best_train_size = 0
        self.best_test_size = 0
        self.best_step_size = 0
        self.best_results = None

        self.test()

        

    

    def update_universe(self, df: pd.DataFrame, max_positions: int = 4) -> pd.Series:
        """
        Updates a DataFrame to track a dynamic universe of coins.
        Should include the dataframe with the lower frequency data. (daily, weekly, etc.)
        Assumes a stacked dataframe
        """
        current_universe = set()
        df['in_universe'] = False

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
        df['in_universe'] = df['in_universe'].shift(periods = 1, freq = '1d')
        self.df = df = df.stack(future_stack = True)
        return df['in_universe'], current_universe

    def strategy(self,
            data,
            params = None,
            ###### To Optimize ######
            std_window = 20,
            mean_window = 20,
            ema_window = 20,
            str_length = 10,
            str_mult = 3,
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
            fixed_tp = True
            ):
        
        if params is not None:
            if isinstance(params, list):
                std_window = params[0]
                mean_window = params[1]
                ema_window = params[2]
                str_length = params[3]
                str_mult = params[4]
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
                str_length = params['str_length']
                str_mult = params['str_mult']
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

        ######################### Signal Generation #########################

        #Generate a signal
        tf = Trend_Following()
        _df = tf.supertrend_signals(data.copy(), str_length, str_mult)

        #Apply tail risk management
        pos = Position(_df, _min_pos, _max_pos)
        _df = pos.initialize_position()
        sl = Stop_Loss(_df, sl_type, sl_ind_length, sl_ind_mult, sl_signal_only)
        _df = sl.apply_stop_loss(fixed_sl, plot = False)
        tp = Take_Profit(_df, tp_type, tp_mult, tp_signal_only)
        _df = tp.apply_take_profit(fixed_tp, plot = False)
        ptp = Take_Profit(_df, tp_type, ptp_mult, ptp_signal_only, exit_percent = ptp_exit_percent)
        _df = ptp.apply_take_profit(fixed_tp, plot = False)

        #Calculate the position size
        _df = cal.merge_cols(_df, common = 'exit_signal', use_clip = True)
        _df = pos.calculate_position(_df)

        #Manage the trade
        mt = Manage_Trade(_df)
        _df = mt.erw_actual_allocation(max_perc_risk, max_dollar_allocation)

        ######################### Universe Selection #########################

        #Update all the columns
        cal = Calculations()
        _df = cal.update_all(_df)

        #Downsample the data
        df = cal.downsample(_df, low_freq)

        #Perform coarse analysis and filtering
        coarse = Coarse()
        df = coarse.volume_flag(df, max_dollar_allocation)
        df = coarse.sort_by_volume(df)
        df = coarse.sort_by_std(df, std_window, mean_window)
        fine = Fine()
        df = fine.above_ema(df, ema_window)

        #apply update_univers
        df['in_universe'], current_universe = self.update_universe(df)

        df.dropna(inplace = True)

        df = df[df['in_universe']]

        return df
    
    def optimize(self):
        """
        Optimize the strategy using the Gaussian Process optimizer

        Returns:
            optimized: The optimized results
        
        !!! Make Sure to Run the test method first to get the best parameters for the optimization !!!

        """
        wfo = WFO(self.df, 
                self.strategy)
        
        best_params = wfo.optimize_parameter_gp(self.best_train_size, self.param_space)
        optimized = wfo.test_strategy(self.best_test_size, best_params)

        return optimized

    
    def test(self) -> None:

        for train_size in range(1000, 3001, 500):  # Adjust the step size as needed
            for test_size in range(1000, 3001, 500):
                for step_size in range(1000, 3001, 500):
                    wfo = WFO(self.df, 
                            self.strategy, 
                            self.param_space, 
                            train_size=train_size, 
                            test_size=test_size, 
                            step_size=step_size, 
                            optimize_fn="gp", 
                            objective='sharpe', 
                            opt_freq='custom')
                    print(f"Train size: {train_size}, Test size: {test_size}, Step size: {step_size}")
                    all_performance, all_results = wfo.walk_forward_optimization()
                    if np.mean(all_performance) > self.best_performance:
                        self.best_performance = np.mean(all_performance)
                        self.best_train_size = train_size
                        self.best_test_size = test_size
                        self.best_step_size = step_size
                        self.best_results = all_results
                    print(f"Mean performance: {np.mean(all_performance)}")

    def stress_test(self):
        pass
    
        