# region imports
from AlgorithmImports import *
from AlphaPerformanceTracker import AlphaPerformanceTracker
# endregion
class MultiAlphaPortfolioConstructionModel(PortfolioConstructionModel):
    def __init__(self, algorithm, rebalance_frequency=30, str_period = 10, str_multiplier = 3, atr_period = 14):
        super().__init__()
        self.algorithm = algorithm
        self.rebalance_frequency = rebalance_frequency
        self.last_rebalance_time = algorithm.Time
        self.alpha_performance_tracker = algorithm.alpha_performance_tracker
        self.tf_allocation = 0.5
        self.mr_allocation = 0.5
        self.str_period = str_period
        self.str_multiplier = str_multiplier
        self.atr_period = atr_period
        self.TF_alpha_models = algorithm.TF_alpha_models
        self.MR_alpha_models = algorithm.MR_alpha_models

    def CreateTargets(self, algorithm, insights: List[Insight]) -> List[PortfolioTarget]:
        # Rebalance monthly
        if self.last_rebalance_time is None or (algorithm.Time - self.last_rebalance_time).days >= self.rebalance_frequency:
            self.Rebalance(algorithm)
            self.last_rebalance_time = algorithm.Time
            self.algorithm.debug(f"Rebalancing...")

        targets = []
        for insight in insights:
            # Calculate target quantity based on allocation and risk
            allocation = self.get_allocation(insight)
            target_quantity = self.calculate_target_quantity(algorithm, insight, allocation)
            targets.append(PortfolioTarget(insight.Symbol, target_quantity))
        return targets

    def Rebalance(self, algorithm):
        # Calculate performance and adjust allocations
        tf_performance, mr_performance = self.alpha_performance_tracker.calculate_category_performance()
        total_cumulative_return = tf_performance + mr_performance

        #Reset total_performance for next rebalancing:
        self.alpha_performance_tracker.total_performance = 0

        
        algorithm.debug(f"Total Cumulative Return for 'TF_Alpha': {total_cumulative_return}")
        algorithm.debug(f"Total Cumulative Return for 'MR_Alpha': {mr_performance}")


        # Softmax for allocation
        total_performance = np.exp(tf_performance) + np.exp(mr_performance)
        
        #Base Cases
        if tf_performance == 0 or mr_performance == 0:
            self.tf_allocation = 0.5
            self.mr_allocation = 0.5
        else:
            self.tf_allocation = np.exp(tf_performance) / total_performance if total_performance != 0 else 0.5
            self.mr_allocation = np.exp(mr_performance) / total_performance if total_performance != 0 else 0.5

    def get_allocation(self, insight):
        if "TF" in insight.SourceModel:
            self.algorithm.debug(f"Allocation for TF_alpha = {self.tf_allocation}")
            return self.tf_allocation
        elif "MR" in insight.SourceModel:
            self.algorithm.debug(f"Allocation for MR_alpha = {self.mr_allocation}")
            return self.mr_allocation
        return 0

    def calculate_target_quantity(self, algorithm, insight, allocation):
        # Calculate the dollar value allocation
        dollar_value_allocation = allocation * algorithm.Portfolio.TotalPortfolioValue

        # Determine the number of source models in the category
        num_models = len(self.TF_alpha_models) if "TF" in insight.SourceModel else len(self.MR_alpha_models)

        # Calculate allocation for each source model
        allocation_per_model = dollar_value_allocation / num_models

        # Calculate max allocation for each insight
        max_allocation_per_insight = allocation_per_model * 0.25

        # Calculate risk based on stop-loss
        current_price = algorithm.Securities[insight.Symbol].Price
        sl_price_level = self.get_stop_loss_level(insight.Symbol, insight.source_model)
        percent_distance = abs(current_price - sl_price_level) / current_price

        # Calculate the actual allocation for the insight
        actual_allocation = max_allocation_per_insight / percent_distance if percent_distance != 0 else 0

        # Calculate target quantity
        target_quantity = actual_allocation / current_price
        return target_quantity

    def get_stop_loss_level(self, symbol, source_model):
        # Define unique stop-loss levels for each symbol
        self._str = self.algorithm.str(symbol, self.str_period, self.str_multiplier , MovingAverageType.Wilders)
        self._atr = self.algorithm.atr(symbol, self.atr_period, MovingAverageType.Simple)

        self.str_value = self._str.current.value
        self.atr_value = self._atr.current.value

        if 'TF' in source_model:
            return self.str_value
        else: #Means 'MR' is in source_model
            return self.atr_value
