from AlgorithmImports import *
from AlphaPerformanceTracker import AlphaPerformanceTracker
from MultiAlphaPortfolioConstructionModel import MultiAlphaPortfolioConstructionModel

class Alpha30DayPerformance(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 12, 31)
        self.SetCash(100000)
        
        # Add universe selection
        self.add_universe(self.MyUniverseSelection)

        # Initialize performance trackers for each alpha model
        self.TF_alpha_models = ['TF_Alpha_1']
        self.MR_alpha_models = ['MR_Alpha_2']
        
        self.alpha_performance_tracker = AlphaPerformanceTracker(self, self.TF_alpha_models + self.MR_alpha_models)

        self.add_alpha(MovingAverageCrossAlphaModel_1(self))
        self.add_alpha(MovingAverageCrossAlphaModel_2(self))

        self.SetPortfolioConstruction(MultiAlphaPortfolioConstructionModel(self))
        # Dictionary to track insights by ID
        self.insight_details = {}
        

    def on_order_event(self, order_event):

        if order_event.status == OrderStatus.FILLED and order_event.Direction == OrderDirection.Sell:
            # Retrieve the order ticket using the order ID from the order event
            ticket = self.transactions.get_order_by_id(order_event.order_id)

            # Ensure the ticket is not None
            if ticket is not None:
                # Access the tag of the order
                order_tag = ticket.tag
                order_symbol = ticket.symbol
                self.alpha_performance_tracker.update_pnl(order_tag, order_symbol, order_event)
                self.debug(f"Order tag: {order_tag}, order_symbol: {order_symbol}")


    def MyUniverseSelection(self, coarse):
        # Select your universe here (e.g., return a list of tickers)
        return [x.Symbol for x in coarse if x.HasFundamentalData and x.Price > 10][:10]

    def OnData(self, data):
        pass
        

    def OnEndOfAlgorithm(self):
        # Get the performance of the specific alphas over the last 30 days
        tf_performance, mr_performance = self.alpha_performance_tracker.calculate_category_performance()

        self.Debug(f"Performance for 'TF_Alpha': {tf_performance}")
        self.Debug(f"Performance for 'MR_Alpha': {mr_performance}")

class MovingAverageCrossAlphaModel_1(AlphaModel):
    def __init__(self, algorithm, short_period=50, long_period=200):
        self.short_period = short_period
        self.long_period = long_period
        self.symbol_data = {}
        self.alpha_performance_tracker = algorithm.alpha_performance_tracker

    def Update(self, algorithm, data):
        insights = []
        for symbol, symbol_data in self.symbol_data.items():
            if not symbol_data.IsReady:
                continue

            short_ma = symbol_data.ShortMA.Current.Value
            long_ma = symbol_data.LongMA.Current.Value

            if short_ma > long_ma and symbol_data.PreviousShortMA <= symbol_data.PreviousLongMA:
                insight = Insight.Price(symbol, timedelta(days=1), InsightDirection.Up)
                insight.SourceModel = 'TF_Alpha_1'
                insights.append(insight)

                """To Add to the main alpha models"""
                self.alpha_performance_tracker.add_insight("TF_Alpha_1", symbol, insight, algorithm.Securities[symbol].Price, algorithm.Time)
                # Place a market order with the insight's SourceModel as a tag
                order = algorithm.MarketOrder(symbol, 1, tag="TF_Alpha_1")
                algorithm.debug(f"Buy Order executed, Tag is: {order.tag}")

            if short_ma < long_ma and symbol_data.PreviousShortMA >= symbol_data.PreviousLongMA:
                order = algorithm.MarketOrder(symbol, -1, tag="TF_Alpha_1")
                algorithm.debug(f"Sell Order executed, Tag is: {order.tag}")

            symbol_data.PreviousShortMA = short_ma
            symbol_data.PreviousLongMA = long_ma

        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = SymbolData(algorithm, symbol, self.short_period, self.long_period)

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.symbol_data:
                del self.symbol_data[symbol]

class MovingAverageCrossAlphaModel_2(AlphaModel):
    def __init__(self, algorithm, short_period=20, long_period=80):
        self.short_period = short_period
        self.long_period = long_period
        self.symbol_data = {}
        self.alpha_performance_tracker = algorithm.alpha_performance_tracker

    def Update(self, algorithm, data):
        insights = []
        for symbol, symbol_data in self.symbol_data.items():
            if not symbol_data.IsReady:
                continue

            short_ma = symbol_data.ShortMA.Current.Value
            long_ma = symbol_data.LongMA.Current.Value

            if short_ma > long_ma and symbol_data.PreviousShortMA <= symbol_data.PreviousLongMA:
                insight = Insight.Price(symbol, timedelta(days=1), InsightDirection.Up)
                insight.SourceModel = 'MR_Alpha_2'
                insights.append(insight)

                self.alpha_performance_tracker.add_insight("MR_Alpha_2", symbol, insight, algorithm.Securities[symbol].Price, algorithm.Time)
                order = algorithm.MarketOrder(symbol, 1, tag= "MR_Alpha_2")
                algorithm.debug(f"Buy Order executed, Tag is: {order.tag}")

            
            if short_ma < long_ma and symbol_data.PreviousShortMA >= symbol_data.PreviousLongMA:
                order = algorithm.MarketOrder(symbol, -1, tag="MR_Alpha_2")
                algorithm.debug(f"Sell Order executed, Tag is: {order.tag}")

            symbol_data.PreviousShortMA = short_ma
            symbol_data.PreviousLongMA = long_ma

        return insights

    def OnSecuritiesChanged(self, algorithm, changes):
        for security in changes.AddedSecurities:
            symbol = security.Symbol
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = SymbolData(algorithm, symbol, self.short_period, self.long_period)

        for security in changes.RemovedSecurities:
            symbol = security.Symbol
            if symbol in self.symbol_data:
                del self.symbol_data[symbol]

class SymbolData:
    def __init__(self, algorithm, symbol, short_period, long_period):
        self.Symbol = symbol
        self.ShortMA = algorithm.SMA(symbol, short_period, Resolution.Daily)
        self.LongMA = algorithm.SMA(symbol, long_period, Resolution.Daily)
        self.PreviousShortMA = 0
        self.PreviousLongMA = 0

    @property
    def IsReady(self):
        return self.ShortMA.IsReady and self.LongMA.IsReady
