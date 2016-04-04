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
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(np.array(timeseries)[:,0], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


# Load data
filename = "AirPassengers.csv"
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv(filename, parse_dates='Month', index_col='Month',date_parser=dateparse)

# Inspect for stationarity
test_stationarity(data)

# Log transform
data_log = np.log(data)
rolmean = data_log.rolling(center=False, window=12).mean()
# Subtract mean and drop NaN
data_log_demeaned = data_log - rolmean
data_log_demeaned.dropna(inplace=True)
test_stationarity(data_log_demeaned)

#Weighted MA, when we don't know the period, adjust parameters when necessary
# http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-moment-functions
expweighted_MA = data_log.ewm(min_periods=0, 
                              adjust=True, 
                              ignore_na=False, 
                              halflife=12).mean()
data_log_ewma = data_log - expweighted_MA
test_stationarity(data_log_ewma)

# Differencing, first order
data_log_diff = data_log - data_log.shift()
plt.plot(data_log_diff)

# Decomposition
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

##########
#Modeling#
##########
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
# AR
model = ARIMA(data_log, order=(2, 1, 0))  
results_AR = model.fit(disp=-1)
RSS = sum(np.array(results_AR.fittedvalues) - np.array(data_log_diff)[:,0]**2)
plt.plot(data_log_diff)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f'% RSS)
# MA
model = ARIMA(data_log, order=(0, 1, 2))  
results_MA = model.fit(disp=-1)
RSS = sum(np.array(results_MA.fittedvalues) - np.array(data_log_diff)[:,0]**2)
plt.plot(data_log_diff)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f'% RSS)
# ARIMA
model = ARIMA(data_log, order=(2, 1, 2))  
results_ARIMA = model.fit(disp=-1)  
RSS = sum(np.array(results_ARIMA.fittedvalues) - np.array(data_log_diff)[:,0]**2)
plt.plot(data_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f'% RSS)
# Get the values back into original scale
preds_ARIMA = pd.Series(results_ARIMA.fittedvalues, copy=True)
preds_ARIMA_cumsum = preds_ARIMA.cumsum()
preds_ARIMA_log = pd.Series(data_log.ix[0].values, index=data_log.index) # base value
preds_ARIMA_log = preds_ARIMA_log.add(preds_ARIMA_cumsum,fill_value=0)
preds_ARIMA = np.exp(preds_ARIMA_log)

plt.plot(data)
plt.plot(preds_ARIMA)
RMSE = np.sqrt(sum((np.array(preds_ARIMA)-np.array(data)[:,0])**2)/len(data))
plt.title('RMSE: %.4f'% np.sqrt(sum((preds_ARIMA-np.array(data)[:,0])**2)/len(data)))


