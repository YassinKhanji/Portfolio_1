import numpy as np


class Costs:
    def __init__(self, df, maker=0.1, taker=0.1):
        """
        Args:
            data (_type_): 
            maker (float, optional):  Defaults to 0.1%.
            taker (float, optional):  Defaults to 0.1%.
        """
        self.df = df
        self.maker = maker
        self.taker = taker
    
    def apply_fees(self, maker = 0.25, taker = 0.40):
        _df = self.df.copy().unstack()


        for coin in _df['close'].columns:
            _df['trades_with_partials', coin] = _df['position', coin].diff().fillna(0)
            _df['trade_costs', coin] = np.where(
                _df['trades_with_partials', coin] > 0,
                _df['trades_with_partials', coin] * (maker / 100),
                _df['trades_with_partials', coin] * (taker / 100) * (-1)
            )
            _df['strategy', coin] = _df['strategy', coin] - _df['trade_costs', coin]
            
        self.df = _df.stack(future_stack=True)
            
        return self.df
    
    def apply_slippage(self):
        pass
