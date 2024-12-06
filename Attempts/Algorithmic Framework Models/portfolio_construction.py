# region imports
from AlgorithmImports import *
# endregion
from datetime import timedelta
class EnergeticLightBrownGaur(QCAlgorithm):
    def initialize(self):
        self.set_start_date(2023, 4, 17)
        self.set_cash(100000)
        self.add_universe(self.my_universe_selection)
        self.add_alpha(EmaCrossAlphaModel())
        self.set_portfolio_construction(ATRBasedPortfolioConstructionModel(self))
        self.set_execution(ImmediateExecutionModel())
        self.enable_automatic_indicator_warm_up = True
    def my_universe_selection(self, coarse):
        return [x.Symbol for x in coarse if x.has_fundamental_data and x.price > 100][:4]
class ATRBasedPortfolioConstructionModel(PortfolioConstructionModel):
    def __init__(self, algorithm, atr_period: int = 14, risk_factor: float = 0.01):
        self.algorithm = algorithm
        self.atr_period = atr_period
        self.risk_factor = risk_factor
        self.atr_indicators = {}
        self.m_r_alpha_models = ['MR']
    def on_securities_changed(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        for security in changes.added_securities:
            if security.symbol not in self.atr_indicators:
                self.atr_indicators[security.symbol] = algorithm.atr(security.symbol, self.atr_period, MovingAverageType.SIMPLE)
                algorithm.warm_up_indicator(security.symbol, self.atr_indicators[security.symbol], Resolution.DAILY)
        for security in changes.removed_securities:
            if security.symbol in self.atr_indicators:
                del self.atr_indicators[security.symbol]
    def create_targets(self, algorithm: QCAlgorithm, insights: List[Insight]) -> List[PortfolioTarget]:
        targets = []
        for insight in insights:
            symbol = insight.symbol
            if symbol not in self.atr_indicators or not self.atr_indicators[symbol].is_ready:
                continue
            atr_value = self.atr_indicators[symbol].current.value
            #self.debug(f"Atr Value: {atr_value}")
            portfolio_value = algorithm.portfolio.total_portfolio_value
            #self.debug(f"Portfolio Value: {portfolio_value}")
            current_price = algorithm.securities[symbol].price
            #self.debug(f"Current Price: {current_price}")
            if current_price==0:
                continue
            target_quantity = self.calculate_target_quantity(symbol, portfolio_value, atr_value, current_price)
            #self.debug(f"Target Quantity: {target_quantity}")
            
            if insight.direction == InsightDirection.UP:
                #self.debug(f"Insight Direction: {insight.direction}")
                targets.append(PortfolioTarget(symbol, target_quantity))
                #self.debug(f"New target Added: {PortfolioTarget(symbol, target_quantity)}")
                
        #self.debug(f"Targets: {targets}")
        return targets
    def calculate_target_quantity(self, symbol, portfolio_value, atr_value, current_price, allocation=0.5):
        dollar_value_allocation = allocation * portfolio_value
        #self.debug(f"Dollar Value Allocation: {dollar_value_allocation}")
        num_models = len(self.m_r_alpha_models)
        allocation_per_model = dollar_value_allocation / num_models
        #self.debug(f"Allocation Per Model: {allocation_per_model}")
        max_allocation_per_insight = allocation_per_model * 0.25
        #self.debug(f"Max Allocation: {max_allocation_per_insight}")
        sl_price_level = current_price - atr_value
        #self.debug(f"Stop Loss Price Level: {sl_price_level}")
        percent_distance = abs(current_price - sl_price_level) / current_price
        if percent_distance * 100 < 1:
            percent_distance = 0.01
        #self.debug(f"Percent Distance: {percent_distance * 100}")
        actual_allocation = max_allocation_per_insight * 0.01 / percent_distance
        target_quantity = actual_allocation / current_price
        #self.debug(f"Portfolio Model; The Target Quantity is {target_quantity} for the symbol {symbol}")
        return target_quantity
