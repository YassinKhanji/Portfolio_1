# region imports
from AlgorithmImports import *
from collections import deque
import numpy as np
# endregion


class AlphaPerformanceTracker:
    def __init__(self, algorithm, alpha_ids, window_size=30):
        self.algorithm = algorithm
        self.alpha_ids = alpha_ids
        self.alpha_insight_pnl = {alpha_id: {} for alpha_id in alpha_ids}  # Track rolling P&L for each alpha
        self.alpha_insight_tracker = {alpha_id: {} for alpha_id in alpha_ids}  # Track insights for each alpha
        self.window_size = window_size  # Rolling window size
        self.flag = False
        self.total_performance = 0

    def add_insight(self, alpha_id, symbol, insight, entry_price, entry_time):
        """Tracks an insight for a specific symbol and alpha."""
        if alpha_id not in self.alpha_insight_tracker:
            self.algorithm.debug(f"Alpha ID {alpha_id} not recognized.")

        if symbol not in self.alpha_insight_tracker[alpha_id]:
            self.alpha_insight_tracker[alpha_id][symbol] = {}
            self.algorithm.debug(f"New symbol added to alpha_insight_tracker: {symbol} withing the alpha_id: {alpha_id}")

        self.alpha_insight_tracker[alpha_id][symbol][insight.id] = {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'direction': insight.Direction
        }

        self.algorithm.debug(f"Insight just added. Entry Price for symbol: {symbol} is {entry_price}")

        # Initialize a P&L deque for the symbol if not already present
        if symbol not in self.alpha_insight_pnl[alpha_id]:
            self.alpha_insight_pnl[alpha_id][symbol] = deque(maxlen=self.window_size)

        self.flag = True

    def update_pnl(self, alpha_id, symbol, orderEvent):
        """Update PnL for an insight based on the order fill for a symbol and alpha."""
        if not self.flag:
            return
        if alpha_id not in self.alpha_insight_tracker:
            self.algorithm.debug(f"Alpha ID: {alpha_id} not recognized.")
            return
        if symbol not in self.alpha_insight_tracker[alpha_id]:
            self.algorithm.debug(f"No insights tracked for symbol: {symbol} in alpha: {alpha_id}")
            return

        for insight_id, insight_info in list(self.alpha_insight_tracker[alpha_id][symbol].items()):
            # Track fill price when an order is filled
            entry_price = insight_info['entry_price']
            return_per_share = orderEvent.FillPrice - entry_price
            total_return = abs(return_per_share * orderEvent.FillQuantity)
            percent_return = total_return / entry_price

        
            self.algorithm.debug(f"Entry Price for symbol: {symbol} is {entry_price}")
            self.algorithm.debug(f"Fill Price for symbol: {symbol} is {orderEvent.FillPrice}")
            self.algorithm.debug(f"Fill Quantity for symbol: {symbol} is {orderEvent.FillQuantity}")
            self.algorithm.debug(f"Total Return for symbol: {symbol} is {total_return}")
            self.algorithm.debug(f"Percent Return for symbol: {symbol} is {percent_return}")
            
            # Add return to the rolling window for this symbol
            self.alpha_insight_pnl[alpha_id][symbol].append(percent_return)
            break

    def calculate_category_performance(self):
        """Calculate the cumulative return, Sharpe ratio, and max drawdown for each alpha."""
        performance_by_alpha = {}
        tf_performance = 0
        mr_performance = 0

        for alpha_id in self.alpha_ids:
            alpha_performance = {}
            self.total_performance = 0  # This will track the performance of the current alpha_id
            
            for symbol, pnl_deque in self.alpha_insight_pnl[alpha_id].items():
                pnl_array = np.array(pnl_deque)
                cumulative_return = np.sum(pnl_array)
                self.total_performance += cumulative_return
                
                sharpe_ratio = np.mean(pnl_array) / np.std(pnl_array) if np.std(pnl_array) != 0 else 0
                # max_drawdown = np.max(np.maximum.accumulate(pnl_array) - pnl_array)
                
                alpha_performance[symbol] = {
                    'cumulative_return': cumulative_return,
                    'sharpe_ratio': sharpe_ratio
                    # 'max_drawdown': max_drawdown
                }
            
            # performance_by_alpha[alpha_id] = alpha_performance
            
            # Update total performance for categories
            if 'TF' in alpha_id:
                tf_performance += self.total_performance
            else:
                mr_performance += self.total_performance

        return tf_performance, mr_performance


    def calculate_cumulative_performance(self, alpha_ids):
        result = self.calculate_category_performance(alpha_ids)
        performance_by_alpha = result[0]
        cumulative_performance = {}

        for alpha_id, performance_by_symbol in performance_by_alpha.items():
            total_cumulative_return = sum(symbol_performance['cumulative_return'] for symbol_performance in performance_by_symbol.values())
            cumulative_performance[alpha_id] = total_cumulative_return

        return cumulative_performance