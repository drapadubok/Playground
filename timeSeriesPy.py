import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
rcParams['figure.figsize'] = 15, 6

def test_stationarity(timeseries):
    http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    #Determing rolling statistics
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



filename = "/m/nbe/scratch/braindata/shared/GraspHyperScan/tempanaconda/AirPassengers.csv"
dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m')
data = pd.read_csv(filename, parse_dates='Month', index_col='Month',date_parser=dateparse)
# Inspect for stationarity
test_stationarity(data)
