import pickle
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def arima_model(train, test, dataset, forecast_weeks, p, d, q):
    """
        Split data, the train, fit, evaluate and plot an ARIMA model for a specified number of weeks ahead to forecast
        
        Args:
            train (dataframe with DatetimeIndex): time series to train model on
            test (dataframe with DatetimeIndex): time series to validate model on
            dataset (string): 'Pre-COVID' or 'Full Dataset'
            forecast_weeks (int): number of weeks ahead to predict
            p (int): the time lag after which partial autocorrelation is no longer significant (aka no longer has an effect on current value). Choose using partial autocorrelation plot.
            d (int): the number of times differencing needs to be applied to achieve stationarity in the time series (aka pass the ADF test)
            q (int): the time lag after which autocorrelation is no longer significant. Choose using autocorrelation plot.
            
        Returns:
            MSE, RMSE, MAE, R2, plot of forecast

    """ 
    warnings.filterwarnings("ignore")
    
    # Instantiate model
    model = ARIMA(train, order=(p,d,q), freq='W-SAT')

    # Fit model
    fitted_model = model.fit()
    print(f'Fitted model summary: \n{fitted_model.summary()}')

    # # Forecast
    forecast = fitted_model.forecast(steps=forecast_weeks)
    for yhat in forecast:
        print(yhat)
    print(f'Forecast: \n{forecast}')

    preds_train = fitted_model.predict(start=train.index[0], end=train.index[-1])
    preds_test = fitted_model.predict(start=test.index[0], end=test.index[-1])

    # plot
    df = pd.concat([train, test], axis=0)
    fig, ax = plt.subplots()
    plt.plot(df.index, df['Total Cases'].values, color='blue', label='Actual Cases')                    # actual cases, all datapoints
    plt.plot(train.index, preds_train, color='green', label='Predicted Cases - Training Set')           # predicted cases, training set
    plt.plot(test.index, preds_test, color='red', label='Predicted Cases - Test Set')                   # predicted cases, test set
    plt.title(dataset + ' Influenza Predictions'+' - '+str(forecast_weeks) +'-Week Forecast')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()
    
    # evaluate
    MSE_train = round(mean_squared_error(train, preds_train),2)
    MSE_test = round(mean_squared_error(test, preds_test),2)
    RMSE_train = round(mean_squared_error(train, preds_train, squared=False),2)
    RMSE_test = round(mean_squared_error(test, preds_test, squared=False),2)
    MAE_train = round(mean_absolute_error(train, preds_train),2)
    MAE_test = round(mean_absolute_error(test, preds_test),2)
    R2_train = r2_score(train, preds_train)
    R2_test = r2_score(test, preds_test)

    results = []

    results.append(MSE_train)
    results.append(MSE_test)
    results.append(RMSE_train)
    results.append(RMSE_test)
    results.append(MAE_train)
    results.append(MAE_test)
    results.append(R2_train)
    results.append(R2_test)
   
    results.insert(0, dataset)
    results.insert(1, forecast_weeks)

    results_matrix = pickle.load(open(r"..\data\results_matrix_ARIMA.pkl", "rb" ))
   
    results_matrix = pd.concat([results_matrix.T, pd.Series(results, index=results_matrix.columns)], axis=1).T

    pickle.dump(results_matrix, open(r"..\data\results_matrix_ARIMA.pkl", "wb" ))

    return results_matrix