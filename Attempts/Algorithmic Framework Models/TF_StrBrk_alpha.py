from AlgorithmImports import *
from datetime import timedelta


class SuperTrendAlphaModel(AlphaModel):
    def __init__(self):
        self.period = 10
        self.multiplier = 3
        self.resolution = Resolution.HOUR
        self.symbol_data_by_symbol = {}
        self.name = f"TF_{self.__class__.__name__}"

    def update(self, algorithm: QCAlgorithm, data: Slice) -> List[Insight]:
        insights = []
        
        for symbol, symbol_data in self.symbol_data_by_symbol.items():
            if not symbol_data.supertrend.is_ready:
                continue

            # Check if the current price is above or below the SuperTrend line
            current_price = data[symbol.value].close
            algorithm.debug(f"{symbol.value}'s Current Price is: {current_price}")
            supertrend_value = symbol_data.supertrend.current.value
            algorithm.debug(f"{symbol.value}'s SuperTrend Current Value is: {supertrend_value}")

            if current_price is not None:
                if current_price > supertrend_value:
                    insights.append(Insight.price(symbol, timedelta(days=1), InsightDirection.UP, source_model= self.name))

                #Might implement the following later
                # elif current_price < supertrend_value:
                #     insights.append(Insight.price(symbol, timedelta(days=1), InsightDirection.DOWN, source_model = self.name))

        return insights

    def on_securities_changed(self, algorithm: QCAlgorithm, changes: SecurityChanges) -> None:
        for added_security in changes.added_securities:
            symbol = added_security.symbol
            if symbol not in self.symbol_data_by_symbol:
                supertrend = SuperTrend(f"Symbol_{symbol}", self.period, self.multiplier, MovingAverageType.WILDERS)
                algorithm.register_indicator(symbol, supertrend, Resolution.HOUR)
                self.symbol_data_by_symbol[symbol] = STRData(supertrend)

        for removed_security in changes.removed_securities:
            symbol = removed_security.symbol
            if symbol in self.symbol_data_by_symbol:
                self.symbol_data_by_symbol.pop(symbol)

class STRData:
    def __init__(self, supertrend: SuperTrend):
        self.supertrend = supertrend
