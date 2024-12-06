from AlgorithmImports import *
from CoinGeckoUniverse import CoinGeckoUniverse
from Criteria import Criteria


class UniverseSelectionModel:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.categories = ['layer-1', 'depin', 'proof-of-work-pow', 'proof-of-stake-pos', 'meme-token', 'dog-themed-coins', 
                           'eth-2-0-staking', 'non-fungible-tokens-nft', 'governance', 'artificial-intelligence', 
                           'infrastructure', 'layer-2', 'zero-knowledge-zk', 'storage', 'oracle', 'bitcoin-fork', 
                           'restaking', 'rollup', 'metaverse', 'privacy-coins', 'layer-0-l0', 'solana-meme-coins', 
                           'data-availabilit', 'internet-of-things-iot', 'frog-themed-coins', 'ai-agents', 
                           'superchain-ecosystem', 'bitcoin-layer-2',  'bridge-governance-tokens', 'modular-blockchain', 
                           'cat-themed-coins', 'cross-chain-communication', 'analytics', 'identity', 'wallets', 
                           'masternodes'] 
        self.coin_universe = CoinGeckoUniverse(self.algorithm, self.categories)
        self.tickers = self.coin_universe.fetch_symbols()
        self.count = 10
        self.std_period = 20
        self.ema_period = 20
        self.criteria_by_symbol = {}

    def coarse_filters(self, coarse):

        for crypto in coarse:
            symbol = crypto.symbol
            
            if symbol.value in self.tickers:
                if symbol not in self.criteria_by_symbol:
                    self.criteria_by_symbol[symbol] = Criteria(self.algorithm, crypto, self.ema_period, self.std_period)
                self.criteria_by_symbol[symbol].update(crypto)

        if self.algorithm.is_warming_up:
            return Universe.UNCHANGED

        ### filter logic using self.criteria_by_symbol
        filtered = [x for x in self.criteria_by_symbol.values() if x._volume_in_usd * 0.05 > self.algorithm.portfolio.total_portfolio_value / (self.algorithm.alpha_counter * 4) and x.criteria_met] 
        self.debug(f"First Symbol in Coarse Filtered: {filtered[0]._value if filtered else 'None'}")

        filtered.sort(key=lambda x: x.percentage_volatility, reverse=True)
        self.debug(f"Sorting Successful: {filtered[0]._value if filtered else 'None'}")

        for x in filtered[:self.count]:
            self.debug('symbol: ' + str(x._value) + '  Volume: ' + str(x._volume_in_usd) + "$")

        self.debug(f"Length of criteria_by_symbol: {len(self.criteria_by_symbol)}")
        self.debug(f"Length of Filtered: {len(filtered)}")
        
        ### return symbols in self.criteria_by_symbol.keys() that pass the filter
        return [f._symbol for f in filtered]

    def debug(self, message):
        if True:  # turn on and off debug
            self.algorithm.Debug(message)
