"""
http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
rcParams['figure.figsize'] = 15, 6


def adfuller_json(ts, autolag="AIC"):
    """
    Wrapper to perform Dickey-Fuller test and return results in json.
    
    Params: 
        ts - a 1d np.array
        autolag - autolag parameter for adfuller, AIC by default
        
    Output:
        res - dict with results
        crit - dict with critical values
        
        prints a table of results as a side effect
    """
    r = adfuller(ts, autolag=autolag)
    res = dict(stat=r[0],
               pval=r[1],
               nlags=r[2],
               nobs=r[3])
               
    neatoutput = pd.Series(r[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    crit = {}
    for key,value in r[4].items():
        crit['crit{0}'.format(key)] = value
        neatoutput['Critical Value ({0})'.format(key)] = value
    print(neatoutput)
    return res, crit
    
    
def test_stationarity(timeseries):    
    #Determing rolling statistics
    #http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    rol = timeseries.rolling(center=False, window=12)
    rolmean = rol.mean()
    rolstd = rol.std()
    
    #Plot rolling statistics:
    plt.plot(timeseries, color='blue',label='Original')
    plt.plot(rolmean, color='red', label='Rolling Mean')
    plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    _,_ = adfuller_json(np.array(timeseries)[:,0])
    
    
def plot_ARIMA_residuals(yhat, orig):
    """
    Plot fitted, original and RSS for an ARIMA model.
    """
    RSS = sum((yhat - orig)**2)
    plt.plot(orig, label='Original data')
    plt.plot(yhat, label='Predicted data', color='red')
    plt.title('RSS: %.4f'% RSS)
    plt.legend(loc='best')
    
    
    
    
# Load data
filename = "AirPassengers.csv"
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv(filename, parse_dates='Month', index_col='Month',date_parser=dateparse)

# Inspect the original data for stationarity
test_stationarity(data)
# See the trend and seasonality

# Log transform
data_log = np.log(data)
test_stationarity(data_log)
# See how std stopped to grow

# Subtract mean and drop NaN
rolmean = data_log.rolling(center=False, window=12).mean()
data_log_demeaned = data_log - rolmean
data_log_demeaned.dropna(inplace=True)
test_stationarity(data_log_demeaned)
# See how the trend disappeared


#Weighted MA, when we don't know the period, adjust parameters when necessary
# http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-moment-functions
expweighted_MA = data_log.ewm(min_periods=0, 
                              adjust=True, 
                              ignore_na=False, 
                              halflife=12).mean()
data_log_ewma = data_log - expweighted_MA
test_stationarity(data_log_ewma)

# Differencing, first order, subtract t+1 from t
data_log_diff = data_log - data_log.shift()
plt.plot(data_log_diff)

# Decomposition, could be very useful but need to understand better how to add back to the forecast
decomposition = seasonal_decompose(data_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid # what is left after removing trend and seasonal
plt.subplot(411)
plt.plot(data_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()
ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)

##########################
#Modeling and forecasting#
##########################
data_log_diff.dropna(inplace=True)
lag_acf = acf(data_log_diff, nlags=20)
lag_pacf = pacf(data_log_diff, nlags=20, method='ols')
#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='gray')
plt.title('Autocorrelation Function')
#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(data_log_diff)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()
# Find where the ACF and PACF curves first hit the upper boundary, use it to select p and q

# AR
model = ARIMA(data_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)
plot_ARIMA_residuals(np.array(results_AR.fittedvalues), np.array(data_log_diff)[:,0])
# MA
model = ARIMA(data_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)
plot_ARIMA_residuals(np.array(results_MA.fittedvalues), np.array(data_log_diff)[:,0])
# ARIMA
model = ARIMA(data_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
plot_ARIMA_residuals(np.array(results_ARIMA.fittedvalues), np.array(data_log_diff)[:,0])

## Get the values back into original scale
# First just take the predicted vals
preds_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)
# Get the cumulative sum
preds_ARIMA_cumsum = preds_ARIMA.cumsum()
# Create a full time series with all values == base value
preds_ARIMA_log = pd.Series(data_log.ix[0].values, index=data_log.index)
# Add the cumsum
preds_ARIMA_log = preds_ARIMA_log.add(preds_ARIMA_cumsum,fill_value=0)
# Exponentiate (remember, we log-transformed the data)
preds_ARIMA = np.exp(preds_ARIMA_log)

plt.plot(data, label="Original data")
plt.plot(preds_ARIMA, label="Predicted data")
RMSE = np.sqrt(sum((np.array(preds_ARIMA)-np.array(data)[:,0])**2)/len(data))
plt.title('RMSE: %.4f'% np.sqrt(sum((preds_ARIMA-np.array(data)[:,0])**2)/len(data)))
plt.legend(loc='best')
