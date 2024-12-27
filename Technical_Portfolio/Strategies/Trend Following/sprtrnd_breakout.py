import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
import os



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Management')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Universe_Selection')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Signal_Generation')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Risk_Management')))

from data import Data
from fetch_symbols import get_symbols
from calculations import Calculations, Metrics
from coarse import Coarse_1 as Coarse
from fine import Fine_1 as Fine
from entry_signal import Trend_Following, Mean_Reversion
from tail_risk import Stop_Loss, Take_Profit
from position_size import Position
from manage_trade import Manage_Trade


class Sprtrnd_Breakout():
    def __init__(self, df):
        self.df = df.copy()

    def update_universe(df: pd.DataFrame, max_positions: int = 4, volume_rank_thr = 50) -> pd.Series:
        """
        Updates a DataFrame to track a dynamic universe of coins.

        df : pd.DataFrame
            A DataFrame with a MultiIndex of (time, coin) and a column 'position'.
            Should already be downsampled to the desired frequency.
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
                    (temp_df['volume_rank'] < volume_rank_thr) &
                    (temp_df['std_rank'] < max_positions) &
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
        length = len(df.index.get_level_values(1).unique())

        
        return df['in_universe'].shift(length), current_universe


    def objective_function(self, df):
        pass

    def optimize(self):
        pass