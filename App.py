import numpy as np
import scipy.stats as st
import pandas as pd
import io
import warnings



  #Make a group of possible Alias'
spy_alias = {"spy_return", "spy", "sp", "benchmark", "mkt", "benchmark_return", "market", "spy_returns"}

returns_alias = {"port", "portfolio", "returns", "portfolio_returns", "return", "data", "strategy", "strategy_return", "port_return","port", "portfolio_returns"}

date_alias = {"date","dates", "time", "date_occured", "datetime", "day", "timestamp"}

def normalize(name):
  return str(name).strip().lower().replace(" ", "_")

def load_returns(input_file):
    #Turn the csv file into an array
  #Make sure everything is the same type.
  if isinstance(input_file, bytes):
    input_file = io.BytesIO(input_file)
  elif isinstance(input_file, str):
    input_file = io.StringIO(input_file)
  df = pd.read_csv("input_file")
  
  lookup = {_normalize(col): col for col in df.columns}

  def _find(aliases, label):
    for alias in aliases:
      if alias in lookup:
        return lookup[alias]

  date_col = _find(date_alias, "date")
  returns_col = _find(returns_alias, "returns")
  spy_col = _find(returns_alias, "spy")

   
def array_split(df):
#convert the 2d array to 1d 

  date = df.("date")numpy_convert
  returns = df.("returns")numpy_convert
  spy = df.df.("spy")numpy_convert
  
  return date, returns, spy

date, returns, spy = array_split(df)
#Now all of our data is in seperate arrays 

#Solve for an array of cumulative returns
def wealth_index (returns):
  initial = float(1)
  total_returns = (1+returns).cumprod()
  wealth_array = initial* (1+returns).cumprod()
  return wealth_array

wealth_array = wealth_index(returns, initial)


#Calculate Drawdown array
def drawdown_fun(returns)
  peak = wealth_array.cummax()
  drawdown = (wealth_array - peak)/peak
  return drawdown

drawdown = drawdown_series(returns)


#Calculate Max Drawdown
def max_drawdown(drawdown):
  drawdown_max = drawdown.min()
  return drawdown_max

drawdown_max = max_drawdown(drawdown)

#calculate number of years
years = len(returns)/252

#Calculate CAGR
def CAGR_fun (returns, years):
  if len(returns) == 0:
    return np.nan
  total_growth = (1+returns).prod()
  if total_growth <= 0:
    return -1.0
  CAGR = total_growth**(1/years)) - 1
  return CAGR

CAGR = CAGR_fun(returns)

excess = returns - rf
sd = sd
var = sd**2

#Calculate Sharpe
def sharpe_fun(returns, rf):

  np.isnan(sd)
  daily_sharpe = (excess.mean())/(sd)
  sharpe_annual = daily_sharpe * (np.sqrt(252))
  return sharpe_annual

sharpe_annual = sharpe_fun(returns, rf)



#Calculate Sortino
def sortino_fun(returns, rf):
  sortino_excess=np.minimum(excess, 0)
  dd = (np.sqrt(np.mean(sortino_excess**2)))
  if dd = 0 or np.isnan(dd):
    return np.nan
  sortino= (np.mean(returns))/dd *np.sqrt(252)
  return sortino

sortino = sortino_fun(returns, rf)


#Make CAPM function
def CAPM_fun(returns, rf, spy, sd):
  #returns - rf = alpha + beta(spy - rf)
  #beta = var(returns)/cov(returns, spy)
  cov = np.cov(returns, spy)
  beta = var/cov



