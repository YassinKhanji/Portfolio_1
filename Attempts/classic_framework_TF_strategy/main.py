# region imports
from AlgorithmImports import *
from CoinGeckoUniverse import CoinGeckoUniverse
# endregion

class GeekyFluorescentPinkPelican(QCAlgorithm):

    def initialize(self):
        self.set_start_date(2020, 1, 1)
        self.set_end_date(2024, 1, 1)
        self.set_cash(100000)
        
        self.set_brokerage_model(BrokerageName.KRAKEN, AccountType.MARGIN)
        self.set_security_initializer(BrokerageModelSecurityInitializer(self.brokerage_model, FuncSecuritySeeder(self.get_last_known_prices)))

        self.set_warm_up(500, Resolution.HOUR)
        self.enable_automatic_indicator_warm_up = True

        self.categories = ['layer-1', 'depin', 'proof-of-work-pow', 'proof-of-stake-pos', 'meme-token', 'dog-themed-coins', 
                           'eth-2-0-staking', 'non-fungible-tokens-nft', 'governance', 'artificial-intelligence', 
                           'infrastructure', 'layer-2', 'zero-knowledge-zk', 'storage', 'oracle', 'bitcoin-fork', 
                           'restaking', 'rollup', 'metaverse', 'privacy-coins', 'layer-0-l0', 'solana-meme-coins', 
                           'data-availabilit', 'internet-of-things-iot', 'frog-themed-coins', 'ai-agents', 
                           'superchain-ecosystem', 'bitcoin-layer-2',  'bridge-governance-tokens', 'modular-blockchain', 
                           'cat-themed-coins', 'cross-chain-communication', 'analytics', 'identity', 'wallets', 
                           'masternodes'] 
        self.coin_universe = CoinGeckoUniverse(self, self.categories)
        # self.symbols = self.coin_universe.fetch_symbols()
        self.symbols = ['BTCUSD']
        self.universe_settings.resolution = Resolution.HOUR
        self.add_universe(CryptoUniverse.kraken(self.crypto_selection))

        self.hourly_symbols = []
        self.symbol_data = {}
        self.averages = {}
        self.brakets = {}

        self.num_models = 1
        self.tp1_mult = 2
        self.tp2_mult = 5

        """--------------------Code Below Not Working (idk why, to review)-----------------------"""

        #     # Schedule universe selection every day at 10:00 AM
        # self.add_universe_selection(ScheduledUniverseSelectionModel(
        #     self.date_rules.every_day(),
        #     self.time_rules.at(0, 0),
        #     CryptoUniverse.kraken(self.crypto_selection)
        # ))

    # def crypto_selection(self, coarse):

    #     # We are going to use a dictionary to refer the object that will keep the moving averages
    #     for c in coarse:
    #         symbol = c.symbol
            
    #         if symbol.value in self.symbols:
    #             if c.symbol not in self.averages:
    #                 self.averages[c.symbol] = SymbolData(c, 480, 480)
    #             self.averages[symbol].update(c.time, c.close)

    #     # Filter the values of the dict: we only want up-trending securities
    #     values = list(filter(lambda x: x.is_uptrend 
    #     and x._volume_in_usd > self.portfolio.total_portfolio_value / (self.num_models * 4),
    #      self.averages.values()))

    #     # Sorts the values of the dict: we want those with greater difference between the moving averages
    #     values.sort(key=lambda x: x.percentage_volatility, reverse=True)

    #     for x in values[:10]:
    #         self.log('Symbol: ' + str(x._symbol.value) + '  Volume in USD: ' + str(x._volume_in_usd))

    #     # we need to return only the symbol objects
    #     values = [ x._symbol.value for x in values ]

    #     return [Symbol.create(value, SecurityType.CRYPTO, Market.KRAKEN) for value in values]

    """---------------------------------------------------------------------"""


    def crypto_selection(self, coarse):

        # We are going to use a dictionary to refer the object that will keep the moving averages
        for c in coarse:
            symbol = c.symbol
            
            if symbol.value in self.symbols:
                if c.symbol not in self.averages:
                    self.averages[c.symbol] = SymbolData(c, 20, 20)
                self.averages[symbol].update(c.time, c.close)

        # Filter the values of the dict: we only want up-trending securities
        values = list(filter(lambda x: x.is_uptrend 
        and x._volume_in_usd > self.portfolio.total_portfolio_value / (self.num_models * 4),
         self.averages.values()))

        # Sorts the values of the dict: we want those with greater difference between the moving averages
        values.sort(key=lambda x: x.percentage_volatility, reverse=True)

        for x in values[:10]:
            self.log('Symbol: ' + str(x._symbol.value) + '  Volume in USD: ' + str(x._volume_in_usd))

        # we need to return only the symbol objects
        return [ x._symbol for x in values ]

    def on_securities_changed(self, changes):
        for added in changes.added_securities:
            symbol = added.symbol
            
            # Subscribe to hourly data only once per symbol
            if symbol not in self.hourly_symbols:
                self.hourly_symbols.append(symbol)
                # Ensure the symbol is subscribed to Hourly data
                # self.add_crypto(symbol.value, Resolution.HOUR, Market.KRAKEN)
            
            if symbol not in self.symbol_data:
                # Now that the symbol has Hourly data, create the Supertrend indicator using Hourly resolution
                self.symbol_data[symbol] = {
                    "supertrend": self.str(symbol, 10, 3, MovingAverageType.WILDERS, Resolution.HOUR),
                    "stop_loss_price": 0,
                    "Percent_distance": 0,
                    "take_profit_price": 0,
                    "partial_take_profit_price": 0,
                    "trailing_stop_price": 0,
                    "partial_take_profit_quantity": 0,
                    "trailing_stop_order_ticket": None,
                    "partial_take_profit_reached": False,
                }

    def on_data(self, slice: Slice):

        for symbol, data in self.symbol_data.items():

            if symbol not in self.hourly_symbols:
                self.debug(f"{self.Time} | {symbol} not present in Hourly Symbols.")
                continue

            if symbol not in slice.bars:
                self.Debug(f"{self.Time} | {symbol} not present in slice.")
                continue

            if self.is_warming_up:
                continue

            supertrend = data["supertrend"]
            if not supertrend.is_ready:
                continue

            current_price = slice[symbol].close
            holdings = self.portfolio[symbol].quantity
            holdings_value = self.portfolio[symbol].holdings_value
            invested = self.portfolio[symbol].invested

            # # Automatically plot the Supertrend indicator
            self.plot_indicator("Supertrend Chart", supertrend)
            self.plot("Supertrend Chart", "ETHUSD", current_price)

            # Check if we should buy
            if supertrend.current.value < current_price and not invested:
                quantity = self.calculate_target_quantity(symbol, self.portfolio.total_portfolio_value, supertrend.current.value, current_price, self.num_models)
                #Place the limit order
                data["stop_loss_price"] = supertrend.current.value
                data["take_profit_price"] = current_price * (1 + data["Percent_distance"] * self.tp2_mult)
                data["partial_take_profit_price"] = current_price * (1 + data["Percent_distance"] * self.tp1_mult)
                data["partial_take_profit_reached"] = False
                data["trailing_stop_order_ticket"] = None
                data['partial_take_profit_quantity'] =  quantity / 2
                self.brakets[symbol] = BraketOrder(self, symbol, quantity, data["stop_loss_price"], data["partial_take_profit_price"], data["take_profit_price"])

    def on_order_event(self, order_event: OrderEvent) -> None:

        if order_event.status != OrderStatus.FILLED:   # only handle filled orders
            self.debug(f"Order Not Filled")
            return

        symbol = order_event.symbol
        order_type = order_event.ticket.order_type
        braket = self.brakets.get(order_event.symbol)
        self.debug(f"Current Order Type: {order_type}")
        self.debug(f"Braket contains: {braket}")
        
        # If stop loss is filled, cancel all orders from this symbol
        if order_type == OrderType.STOP_MARKET:
            self.transactions.cancel_open_orders(symbol, "Hit stop price")

        # If take profit is filled, cancel all orders after the 2nd stop loss is filled.
        if order_type == OrderType.LIMIT:
            if order_event.order_id == braket.tp1.order_id:
                braket.sl.cancel()
                # self.trailing_stop_order(
                #         symbol, (-self.portfolio.cash_book[symbol.value[:-3]].amount),
                #         stop_price = self.symbol_data[symbol]['stop_loss_price'],
                #         trailing_amount = self.symbol_data[symbol]['stop_loss_price'],
                #         trailing_as_percentage=False
                #     )
                self.debug(f"Hit 1st take profit")
            if order_event.order_id == braket.tp2.order_id:
                self.transactions.cancel_open_orders(symbol, "Hit 2nd take profit")


    def calculate_target_quantity(self, symbol, portfolio_value, str_value, close_price, num_models):
        allocation = 1 / num_models

        dollar_value_allocation = allocation * portfolio_value
        self.debug(f"Dollar Value Allocation: {dollar_value_allocation}")

        allocation_per_model = dollar_value_allocation / num_models
        self.debug(f"Allocation Per Model: {allocation_per_model}")

        max_allocation_per_insight = allocation_per_model * 0.25
        self.debug(f"Max Allocation: {max_allocation_per_insight}")

        sl_price_level = str_value
        self.sl_price_level = sl_price_level
        self.debug(f"Stop Loss Price Level: {sl_price_level} for symbol: {symbol}")

        percent_distance = (close_price - sl_price_level) / close_price
        self.symbol_data[symbol]["Percent_distance"] = percent_distance * 0.25
        if percent_distance * 100 < 1:
            percent_distance = 0.01
        self.debug(f"Percent Distance: {percent_distance * 100}")

        actual_allocation = max_allocation_per_insight * 0.01 / percent_distance
        self.debug(f"Actual Allocation: {actual_allocation}")
        target_quantity = actual_allocation / close_price
        self.debug(f"Portfolio Model; The Target Quantity is {target_quantity} for the symbol {symbol}")

        return target_quantity

class SymbolData(object):
    def __init__(self, crypto, ema_period, std_period):
        self._symbol = crypto.symbol
        self._volume_in_usd = crypto.volume_in_usd
        self._std = StandardDeviation(std_period)
        self._ema = ExponentialMovingAverage(ema_period)
        self.percentage_volatility = 0
        self.is_uptrend = False
        self.scale = 0

    def update(self, time, value):
        if self._std.update(time, value) and self._ema.update(time, value):
            self.percentage_volatility = (self._std.current.value / value) * 100
            self.is_uptrend = value > self._ema.current.value

class BraketOrder:
    def __init__(self, algorithm, symbol, quantity, stop_price, limit_price_1, limit_price_2):
        # calculate quantity, stop price, limit prices
        algorithm.market_order(symbol, quantity)
        self.sl = algorithm.stop_limit_order(symbol, -algorithm.portfolio.cash_book[symbol.value[:-3]].amount, stop_price, stop_price * 1.001)
        self.tp1 = algorithm.limit_order(symbol, -algorithm.portfolio.cash_book[symbol.value[:-3]].amount/2, limit_price_1)
        self.tp2 = algorithm.limit_order(symbol, -algorithm.portfolio.cash_book[symbol.value[:-3]].amount, limit_price_2)

