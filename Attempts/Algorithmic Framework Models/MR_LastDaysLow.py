from AlgorithmImports import *
class LastDaysLow(AlphaModel):
    def __init__(self, algorithm):
        self.symbol_data = {}
        self.name = f"MR_{self.__class__.__name__}"
        self.alpha_performance_trackers = algorithm.alpha_performance_trackers

    def Update(self, algorithm, data):
        insights = []
        for symbol, symbol_data in self.symbol_data.items():
            if symbol not in data:
                continue

            # Ensure we have enough data
            if symbol_data.daily_lows.IsReady and symbol_data.hourly_closes.IsReady:
                yesterdays_low = symbol_data.daily_lows[0]
                b4_yesterdays_low = symbol_data.daily_lows[1]
                last_candle_close = symbol_data.hourly_closes[1]
                current_candle_close = symbol_data.hourly_closes[0]
                last_candle_open = symbol_data.hourly_opens[1]

                # Generate insight if conditions are met
                criteria_1 = last_candle_close < yesterdays_low and last_candle_open > yesterdays_low and current_candle_close > yesterdays_low
                criteria_2 = last_candle_close < b4_yesterdays_low and last_candle_open > b4_yesterdays_low and current_candle_close > b4_yesterdays_low
                if criteria_1 or criteria_2:
                    insights.append(Insight.Price(symbol, timedelta(hours=1), InsightDirection.Up, source_model = self.name))
                    algorithm.debug(f"Insight generated. Hourly Close: {current_candle_close}, Yesterday's Low: {yesterdays_low}, Last Hourly Close: {last_candle_close}")


        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = SymbolData(algorithm, symbol)

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.symbol_data:
                del self.symbol_data[symbol]

class SymbolData:
    def __init__(self, algorithm, symbol):
        self.symbol = symbol
        self.daily_lows = RollingWindow[float](2)
        self.hourly_closes = RollingWindow[float](2)
        self.hourly_opens = RollingWindow[float](2)

        # Subscribe to daily and hourly consolidators
        algorithm.Consolidate(symbol, Resolution.Daily, self.OnDailyData)
        algorithm.Consolidate(symbol, Resolution.Hour, self.OnHourlyData)

    def OnDailyData(self, bar):
        self.daily_lows.Add(bar.Low)

    def OnHourlyData(self, bar):
        self.hourly_closes.Add(bar.Close)
        self.hourly_opens.add(bar.Open)
