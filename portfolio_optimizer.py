#!python3

#Import the modules needed
import bs4 as bs
import datetime as dt 
import os
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 
import pandas as pd 
import pandas_datareader.data as web
import pickle
import requests
import sys

print(sys.stdin)

# Set style
style.use('ggplot')

# CREATE FUNCTIONS
# Define and save selected stocks
def save_sp500_tickers():
	tickers = ['CNP', 'F', 'WMT', 'GE', 'TSLA']

	with open("sp500tickers.pickle", "wb") as f:
		pickle.dump(tickers, f)

	print(tickers)

	return tickers

#save_sp500_tickers()

# Get stock data from yahoo finance
def get_data_from_yahoo(reload_sp500 = False):
	if reload_sp500:
		tickers = save_sp500_tickers()
	else:
		with open("sp500tickers.pickle", "rb") as f:
			tickers = pickle.load(f)

	if not os.path.exists('stock_dfs'):
		os.makedirs('stock_dfs')

	start = dt.datetime(2014, 1, 1)
	end = dt.datetime(2018, 2, 2)

	for ticker in tickers: #for ticker in tickers: <== if you want the 500 tickers or for ticker in tickers[:10]: <== if you want the 10 tickers
		print(ticker)
		if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
			df = web.DataReader(ticker, 'yahoo', start, end)
			df.to_csv('stock_dfs/{}.csv'.format(ticker))

		else:
			print('Already have {}'.format(ticker))

#get_data_from_yahoo()

# Put Adjusted close prices in a single file
def compile_data():
	with open("sp500tickers.pickle", "rb") as f:
		tickers = pickle.load(f)
	
	main_df = pd.DataFrame()

	for count, ticker in enumerate(tickers):
		df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
		df.set_index('Date', inplace = True)

		df.rename( columns = {'Adj Close': ticker}, inplace = True)
		df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace = True)

		if main_df.empty:
			main_df = df
		else:
			main_df = main_df.join(df, how = 'outer')

		if count % 10 == 0:
			print(count)

	print(main_df.head())
	main_df.to_csv('sp500_joined_closes.csv')

#compile_data()

# Create a function to calculate returns
def calculate_returns():
	main_df = pd.read_csv('sp500_joined_closes.csv')
	main_df.set_index('Date', inplace = True)


	returns_daily = main_df.pct_change()
	returns_daily.to_csv('sp500_daily_returns.csv')
	returns_annual = returns_daily.mean() * 250

	#print(returns_daily.head())
	#print(returns_annual)

	return returns_daily, returns_annual

#calculate_returns()


# OPTIMIZE PORTFOLIO
# Create a function to calculate covariance
def calculate_covariance():

	ret_daily, ret_annual = calculate_returns()

	covariance_daily = ret_daily.cov()
	covariance_daily.to_csv('sp500_daily_covariance.csv')
	covariance_annual = covariance_daily * 250
	covariance_annual.to_csv('sp500_annual_covariance.csv')

	#print(covariance_daily.head())
	#print(covariance_annual)

	return covariance_daily, covariance_annual

#calculate_covariance()

# Create empty lists to store returns, volatility and weights for simulated portfolios
port_returns = []
port_volatility = []
stock_weights = []
sharpe_ratio = []

# Extract returns and covariances from calculate_returns() and calculate_covariance()
ret_daily, ret_annual = calculate_returns()
cov_daily, cov_annual = calculate_covariance()

# Set number of assets and number of simulated portfolios
tickers_selected = save_sp500_tickers()
num_assets = len(tickers_selected)
num_portfolios = 50000


# Populate the empty lists with each portfolio returns,risk and weights
for single_portfolio in range(num_portfolios):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    returns = np.dot(weights, ret_annual)
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_annual, weights)))
    sharpe = returns / volatility
    sharpe_ratio.append(sharpe)
    port_returns.append(returns)
    port_volatility.append(volatility)
    stock_weights.append(weights)

# Create a dictionary for Returns and Risk values of each portfolio
portfolio = {'Returns': port_returns,
             'Volatility': port_volatility,
             'Sharpe Ratio': sharpe_ratio}

# Accomodate each ticker and weight in the portfolio
for counter,symbol in enumerate(tickers_selected):
    portfolio[symbol+' Weight'] = [Weight[counter] for Weight in stock_weights]

# Enhance dataframe
df = pd.DataFrame(portfolio)
df.to_csv('sp500_portfolio.csv')

# get better labels for desired arrangement of columns
column_order = ['Returns', 'Volatility', 'Sharpe Ratio'] + [stock+' Weight' for stock in tickers_selected]

# reorder dataframe columns
df = df[column_order]

# Find the optimal Sharpe Ratio and the lowest volatility portfolio
lowest_volatility = df['Volatility'].min()
highest_sharpe_ratio = df['Sharpe Ratio'].max()

# use the min, max values to locate and create the two special portfolios
sharpe_portfolio = df.loc[df['Sharpe Ratio'] == highest_sharpe_ratio]
min_variance_port = df.loc[df['Volatility'] == lowest_volatility]

# print the details of the 2 efficient portfolios
MinVarPort = min_variance_port.T
SharpePort = sharpe_portfolio.T
print(MinVarPort)
print(SharpePort)
MinVarPort.to_csv('sp500_min_variance.csv')
SharpePort.to_csv('sp500_sharpe_ratio.csv')


# Plot the efficient frontier with a scatter plot
plt.style.use('seaborn-dark')
df.plot.scatter(x ='Volatility', y ='Returns', c = 'Sharpe Ratio', cmap = 'RdYlGn', edgecolors = 'black', figsize=(10, 8), grid=True)
plt.scatter(x=sharpe_portfolio['Volatility'], y=sharpe_portfolio['Returns'], c='red', marker='D', s=200)
plt.scatter(x=min_variance_port['Volatility'], y=min_variance_port['Returns'], c='blue', marker='D', s=200 )
plt.xlabel('Volatility (Std. Deviation)')
plt.ylabel('Expected Returns')
plt.title('Efficient Frontier')
plt.show()



