import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def VaR(returns, cl=5):
    return 1-np.percentile(returns, cl)
    
def CVaR(returns, cl=5):
    belowvar=returns < (1-VaR(returns,cl))
    return 1-returns[belowvar].mean()
    
def Simulate(tickers, weights, mean_returns, covmatrix, n_sims=1000, n_days=252, initialval=10000):

    mean_mat=np.full(shape = (n_days, len(weights)), fill_value = mean_returns)
    portfolio_sims = np.full(shape = (n_days, n_sims), fill_value=0.0)
    L = np.linalg.cholesky(covmatrix)
    
    for i in range(n_sims):
        Z=np.random.normal(size=(n_days,len(weights)))
        daily_returns = mean_mat + np.dot(L,Z.T).T
        port_returns = np.dot(daily_returns,weights) + 1
        portfolio_sims[:,i] = np.cumprod(port_returns)*initialval
    end_returns=portfolio_sims[-1,:]/initialval
    simVaR=VaR(end_returns)
    simCVaR=CVaR(end_returns)
    simVaRval=(1-simVaR)*initialval
    simCVaRval=(1-simCVaR)*initialval
    
    plt.plot(portfolio_sims)
    plt.xlabel("Market Days")
    plt.ylabel("Portfolio Value")
    plt.title("Monte Carlo Simulation of Portfolio Performance")
    plt.axhline(simVaRval, color='red', linestyle='--', label=f"VaR : {simVaR*100:.2f}%")
    plt.axhline(simCVaRval, color='green', linestyle='--', label=f"CVaR : {simCVaR*100:.2f}%")
    plt.legend()
    plt.show()





