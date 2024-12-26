import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from itertools import product

import requests
import json

import scipy.stats as stats
from scipy.optimize import minimize

import quantstats as qs

class Sprtrnd_Breakout():
    def __init__(self, df):
        self.df = df.copy()

    def optimize(self, df):
        pass

    def objective_function(self):
        pass