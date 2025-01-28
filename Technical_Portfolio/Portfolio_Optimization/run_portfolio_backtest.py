import requests
import json
import math
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
# from qgridnext import show_grid
from datetime import datetime, timedelta
import pickle
import sys  
import os
import pandas_ta as ta
import sklearn as sk
import datetime as dt
from skopt.space import Integer, Real, Categorical
import random

# Ensure the directories are in the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data_Management')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Strategies', 'Trend_Following')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Strategies', 'Mean_Reversion')))


# Import the modules
from data import Data, get_symbols_for_bot
from sprtrnd_breakout import Sprtrnd_Breakout
from last_days_low import Last_Days_Low
from portfolio_optimization import Portfolio_Optimization
from portfolio_risk_management import Portfolio_RM


start_time = dt.datetime(2020, 1, 1)
end_time = dt.datetime(2025, 1, 21)
timeframes = ['1w', '1d', '4h', '1h', '30m','15m', '5m', '1m']
index = 3 #It is better to choose the highest frequency for the backtest to be able to downsample
interval = timeframes[index]
symbols = ['MANAUSD','BONKUSD','BANDUSD','PHAUSD','POLUSD','STORJUSD','ETHUSD','SCUSD',
                'ATOMUSD','RLCUSD','GMTUSD','LTCUSD','DYMUSD','SEIUSD','QTUMUSD','MASKUSD','CTSIUSD',
                'TONUSD','OPUSD','ARKMUSD','FORTHUSD','CHRUSD','RUNEUSD','ZROUSD','ENJUSD','SAGAUSD',
                'ZECUSD','ENSUSD','SUIUSD','SHIBUSD','ETHFIUSD','CELRUSD','NEIROUSD',
                'ZKUSD','APTUSD','LINKUSD','ICXUSD','APEUSD','EGLDUSD','API3USD','DASHUSD','STRKUSD','ICPUSD',
                'SANDUSD','FLOWUSD','ALTUSD','MINAUSD','TURBOUSD','CVCUSD','FETUSD','JASMYUSD','RENDERUSD','OGNUSD',
                'NEARUSD','COTIUSD','STGUSD','IMXUSD','WIFUSD','DOTUSD','GRTUSD','SYNUSD','MEMEUSD','PEPEUSD','LSKUSD',
                'AVAXUSD','LDOUSD','BTCUSD','FXSUSD','TAOUSD','LUNAUSD','BCHUSD','LPTUSD','AUDIOUSD','MOVRUSD','ETCUSD',
                'ADAUSD','STXUSD','LRCUSD','FTMUSD','KSMUSD','FILUSD',
                'EIGENUSD','PONDUSD','RAREUSD','PNUTUSD','OMNIUSD','ALGOUSD','ANKRUSD','TRXUSD','DENTUSD',
                'XTZUSD','DOGEUSD','OXTUSD','SOLUSD','ZRXUSD','GLMRUSD','ARBUSD','TIAUSD','FIDAUSD','RADUSD',
                'BLURUSD', 'XRPUSD']
# symbols = ['BTCUSD', 'ETHUSD']
symbols = random.sample(symbols, 20)
print(symbols)
data = Data(symbols, interval, start_time, end_time).df

mr_strat_1 = Last_Days_Low(data, objective = 'multiple', train_size = 500, test_size = 500, step_size = 500)
tf_strat_1 = Sprtrnd_Breakout(data, objective = 'multiple', train_size = 500, test_size = 500, step_size = 500)

mr_strat_1.test()
tf_strat_1.test()


#Create a dummy results that represents holding cash where the value of the portfolio is constant
cash_df = pd.DataFrame(data = {'strategy': np.zeros(data.shape[0]), 'portfolio_value': np.ones(data.shape[0])}, index = data.index)
cash_df
strategy_map = {'mr_strat_1': mr_strat_1.results.strategy,
                'tf_strat_1': tf_strat_1.results.strategy}

portfolio_wfo = Portfolio_Optimization(strategy_map, train_size = 500, test_size = 500, step_size = 500, objective = 'multiple')
all_performance, all_results = portfolio_wfo.walk_forward_optimization()
print(all_results)
all_results = all_results.fillna(0)
results_creturns = all_results.cumsum().apply(np.exp)
print(all_results[0])
portfolio_rm = Portfolio_RM(all_results[0])
drawdown_limit, in_drawdown = portfolio_rm.drawdown_limit(-0.15)
print(drawdown_limit)
creturns = drawdown_limit.cumsum().apply(np.exp)

plt.figure(figsize=(10, 6))
plt.plot(results_creturns.reset_index(drop=True), label='Strategy')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.title('Portfolio Results Before Portfolio Risk Management')
plt.legend()
plt.grid()
plt.savefig("portfolio_plot_b4_rm.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(creturns.reset_index(drop=True), label='Strategy')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.title('Portfolio Results')
plt.legend()
plt.grid()
plt.savefig("portfolio_plot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(mr_strat_1.results.cstrategy.reset_index(drop=True), label='Strategy')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.title('MR_Strat_1 Results')
plt.legend()
plt.grid()
plt.savefig("first_plot.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(tf_strat_1.results.cstrategy.reset_index(drop=True), label='Strategy')
plt.xlabel('Time')
plt.ylabel('Performance')
plt.title('TF_Strat_1 Results')
plt.legend()
plt.grid()
plt.savefig("second_plot.png")
plt.close()

with open("variables.pkl", "wb") as f:  # Use 'wb' mode for writing in binary
    pickle.dump((mr_strat_1.results, tf_strat_1.results, strategy_map, all_performance, all_results, drawdown_limit, in_drawdown, creturns), f)