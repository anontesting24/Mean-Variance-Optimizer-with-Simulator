import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from MonteCarlo import Simulate

def std_deviation(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def mean_return(weights, mean_returns):
    return (1+np.sum(mean_returns * weights))**252-1

def sharpe_ratio(weights, mean_returns, cov_matrix, rf_rate):
    return (mean_return(weights, mean_returns) - rf_rate) / std_deviation(weights, cov_matrix)

def negative_sr(weights, returns, cov_matrix, rf_rate):
    return -sharpe_ratio(weights, returns, cov_matrix, rf_rate)

class mvo:
    def __init__(self):
        self.tickers=set()
        self.tickerlist=[]
        self.weights=None
        self.returns=None
        self.mean_returns=None  #daily
        self.cov_matrix=None  #annulized
        self.rf_rate=None
        self.sr=None

    def add_tickers(self, tickers=[]):
        if not tickers:
            return
        for i in tickers:
            self.tickers.add(i)
        self.weights=None
        
    def remove_tickers(self, tickers=[]):
        if not tickers:
            return
        for i in tickers:
            self.tickers.remove(i)
        self.weights=None
            
    def empty_tickers(self):
        self.tickers.clear()
        self.weights=None

    
    def results(self):
        if self.weights is None:
            print("kindly optimize again")
            return
        print("Optimal Weights:")
        for ticker, weight in zip(self.tickerlist, self.weights):
            print(f"{ticker}: {weight:.4f}")

        optimal_portfolio_return = mean_return(self.weights, self.mean_returns)
        optimal_portfolio_volatility = std_deviation(self.weights, self.cov_matrix)
        optimal_sharpe_ratio = sharpe_ratio(self.weights, self.mean_returns, self.cov_matrix, self.rf_rate)
        
        print(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
        print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
        print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

        
    def optimize(self, years, lower_bound,upper_bound,rf_rate=None):
        if rf_rate==None:
            rfrate= yf.download('^IRX', period="1d", interval="1d")
            rfrate = rfrate['Close']
            rf_rate = rfrate.iloc[-1,-1]/100

        end_date= datetime.today()
        start_date = end_date - timedelta(days=years*365)

        adjclose_df = pd.DataFrame()
        self.tickerlist=list(self.tickers)
        adjclose_df=yf.download(self.tickerlist, start = start_date, end = end_date)['Close']
        adjclose_df=adjclose_df[self.tickerlist]
        adjclose_df=adjclose_df.dropna()
            
        returns=adjclose_df/adjclose_df.shift(1) - 1
        # returns=np.log(adjclose_df/adjclose_df.shift(1)) #for lognormal returns
        returns=returns.dropna()
        self.cov_matrix=returns.cov()*252
        self.mean_returns=returns.mean()
        self.rf_rate=rf_rate
        
        constraints = {'type' : 'eq', 'fun' : lambda weights: np.sum(weights) - 1}
        bounds = [(lower_bound,upper_bound) for _ in range(len(self.tickers))]
        initial_weights = np.array([1/len(self.tickers)]*len(self.tickers))
        # initial_weights=np.array([0.3,0.1,0.15,0.24,0.21])
        optimized_result = minimize(negative_sr, initial_weights, args=(self.mean_returns, self.cov_matrix, rf_rate), method='SLSQP', constraints=constraints, bounds=bounds)
        optimal_weights = optimized_result.x
        self.sr = -optimized_result.fun
        self.weights = optimal_weights

        self.results()
    def MonteCarloSim(self, n_sims = 1000, n_days = 252, initialval = 10000 ):
        Simulate(self.tickerlist,self.weights,self.mean_returns,self.cov_matrix/252, n_sims, n_days, initialval)
        
        
        
        
