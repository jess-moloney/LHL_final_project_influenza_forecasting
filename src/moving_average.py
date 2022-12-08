import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import mean
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def moving_average(data, dataset, train_test, window, forecast_weeks):
    """
        Calculate and plot the moving average for a specified number of weeks ahead to forecast
        
        Args:
            data (dataframe with DatetimeIndex): time series to calculate moving average for
            dataset (string): 'Pre-COVID' or 'Full Dataset'
            train_test (string): 'Training' or 'Test'
            window (int): Moving average window
            forecast_weeks (int): number of weeks ahead to predict
            
        Returns:
            MSE, RMSE, MAE, R2, plot of forecast

    """ 
    if forecast_weeks > 1:
        freq = str(forecast_weeks)+'W'
        data = data.set_index('Week Ending')
        data = data.resample(freq).mean()
   
    X = data['Total Cases']
    window = window
    history = [X[i] for i in range(window)]
    test = [X[i] for i in range(window, len(X))]
    predictions = list()
   
    # walk forward over time steps in test
    for t in range(len(test)):
        length = len(history)
        yhat = mean([history[i] for i in range(length-window, length)])
        obs = test[t]
        predictions.append(yhat)
        history.append(obs)
        # print(f'predicted={yhat}, actual={obs}')
   
    # plot
    fig, ax = plt.subplots()
    plt.plot(test, color='blue', label='Actual Cases')
    plt.plot(predictions, color='red', label='Predicted Cases')
    plt.title(dataset + ' Cases Prediction'+' - '+str(forecast_weeks) +'-Week Forecast')
    plt.xlabel('Date')
    plt.ylabel('Cases')
    plt.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.show()
   
    if train_test == 'Training':
        MSE_train = round(mean_squared_error(X[window:], predictions),2)
        RMSE_train = round(mean_squared_error(X[window:], predictions, squared=False),2)
        MAE_train = round(mean_absolute_error(X[window:], predictions),2)
        R2_train = r2_score(X[window:], predictions)
        MSE_test = np.nan
        RMSE_test = np.nan
        MAE_test = np.nan
        R2_test = np.nan
    else:
        MSE_test = round(mean_squared_error(X[window:], predictions),2)
        RMSE_test = round(mean_squared_error(X[window:], predictions, squared=False),2)
        MAE_test = round(mean_absolute_error(X[window:], predictions),2)
        R2_test = r2_score(X[window:], predictions)
        MSE_train = np.nan
        RMSE_train = np.nan
        MAE_train = np.nan
        R2_train = np.nan
   
    results = []
   
    results.append(MSE_train)
    results.append(MSE_test)
    results.append(RMSE_train)
    results.append(RMSE_test)
    results.append(MAE_train)
    results.append(MAE_test)
    results.append(R2_train)
    results.append(R2_test)
   
    results.insert(0, train_test)
    results.insert(1, dataset)
    results.insert(2, forecast_weeks)
   
    results_matrix = pickle.load(open(r"..\data\results_matrix_moving_average.pkl", "rb" ))
   
    results_matrix = pd.concat([results_matrix.T, pd.Series(results, index=results_matrix.columns)], axis=1).T
   
    pickle.dump(results_matrix, open(r"..\data\results_matrix_moving_average.pkl", "wb" ))
   
    return results_matrix