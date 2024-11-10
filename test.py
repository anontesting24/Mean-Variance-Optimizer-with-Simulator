import MVO

test_portfolio=MVO.mvo()

test_portfolio.add_tickers(['GLD', 'QQQ', 'SPY', 'VTI', '^NSEI'])

test_portfolio.optimize(5,0,0.5)

test_portfolio.MonteCarloSim(100,100,1000)





