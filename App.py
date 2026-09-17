import numpy as np
import scipy.stats as st
import pandas as pd
import io
import warnings
import statsmodels.api as sm
import getFamaFrenchFactors as gff


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
  df = pd.read_csv(input_file)
  
  lookup = {normalize(col): col for col in df.columns}
  
  #check for alias names in list. else raise error
  def find(aliases, label):
    for alias in aliases:
      if alias in lookup:
        return lookup[alias]
      raise ValueError("Needs relevant column titles (Example: dates, returns, spy)")

  date_col = find(date_alias, "date")
  returns_col = find(returns_alias, "returns")
  spy_col = find(spy_alias, "spy")

  
  #Turn all of your data to numbers, if theres any strings it gets turned into np.nan
  clean_df = pd.DataFrame({
    "returns":pd.to_numeric(df[returns_col], errors = "coerce"),
    "spy" :pd.to_numeric(df[spy_col], errors= "coerce")
  })
  #turn dates into datetime
  clean_df.index = pd.to_datetime(df[date_col], errors="coerce")

  #gets rid of "NA"s and puts it in order
  clean_df = clean_df[~clean_df.index.isna()].dropna().sort_index()

  if len(clean_df) < 30:
    raise ValueError(f"Needs at least 30 valid days; found {len(clean_df)}.")
    
  start = clean_df.index.min()
  end = clean_df.index.max()
  
  return clean_df, start, end

 
clean_df, start, end = load_returns(input_file)

  
#Fama french factors, also has risk free rate(rf)
def fama_french(start, end):
  factors = gff.famaFrench3Factor(frequency="d")

  #Change date column into datetime format under pd
  factors["date"] = pd.to_datetime(factors["date_ff_factors"])
  factors = factors.set_index("date").sort_index()
  #isolate date segment
  factors_slice = factors.loc[start:end]
  #extraction
  rf = factors_sliced["RF"]
  mkt_rf = factors_sliced["Mkt-Rf"]
  smb = factors_sliced["SMB"]
  hml = factors_sliced["HML"]
  return factors_sliced, mkt_rf, smb, hml, rf

factors_df, mkt_rf, smb, hml, rf = fama_french(start, end)

def fama_french_regression(returns, factors_df):
  #Merge the two dataframes so everthing is synced for convenience.
  merged = pd.concat([returns.rename("returns"), factors_df], axis = 1, join = "inner").dropna()
  excessy = merged["returns"] - merged["RF"]
  excessx = merged["SPY"] - merged["Mkt-RF"]
  #Pull out variables we'll show
  X = merged[["Mkt-RF", "SMB", "HML"]]

  #Original model:
  #y = a + (b1x1) + (b2x2) + (b3x3).

  x_const = sm.add_constant(x)
#set up least squares problem. the x const we added makes sure that alpha has a coefficient of 1 and not 0. .fit() performs matrix algebra of (X^T*X)^-1 * X^T * y to calculate best fit coefficients
  model = sm.OLS(y, x_const).fit()

  alpha = float(model.params["const"] * 252
  beta_mkt = float(model.params["Mkt-RF"] 
  beta_smb = float(model.params["SMB"]  
  beta_hml = float(model.params["HML"]  
  r_squared = float(model.rsquared)



#Solve for an array of cumulative returns
def wealth_index (returns):
  initial = float(1)
  total_returns = (1+returns).cumprod()
  wealth_array = initial* (1+returns).cumprod()
  return wealth_array

wealth_array = wealth_index(returns)


#Calculate Drawdown array
def drawdown_fun(returns):
  peak = returns.cummax()
  drawdown = (returns - peak)/peak
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
  if dd == 0 or np.isnan(dd):
    return np.nan
  sortino= (np.mean(returns))/dd *np.sqrt(252)
  return sortino

sortino = sortino_fun(returns, rf)


#Make CAPM function
def CAPM_regression(returns,spy,rf):

  #calculate returns-rf
  y = returns - rf
  x = spy - rf
  beta = (y.cov(x))/x.var()
  daily_alpha = y.mean() - (beta * x.mean())
  alpha= daily_alpha *252
  R = y.corr(x)
  R_square = R ** 2
  return beta,alpha,R_square





