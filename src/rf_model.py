import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
seed_value = 2022

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
plt.rcParams['figure.figsize'] = [20, 10]

import sys
sys.path.append(r"C:\Users\User\Documents\projects\LHL_final_project_influenza_forecasting\src")
from evaluate_model import *
from retrieve_name import *

def rf_model(data, weeks_to_predict, df_name, max_depth=None):
    """
        Split data, then train, fit, evaluate and plot an Random Forest model for a specified number of weeks ahead to forecast
        
        Args:
            data (dataframe with DatetimeIndex): time series to train and validate model on
            weeks_to_predict (int): number of weeks ahead to predict
            max_depth (int): The maximum depth of the tree. If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
            
        Returns:
            printout of shape of training X and y, shape of test X and y, plot of forecast, result_matrix with MSE, RMSE, MAE for train and test sets appended

    """ 
    # split into train and test
    X = data.loc[:, data.columns != 'Total Cases']
    y = data['Total Cases']

    if weeks_to_predict == 1:
        train_X = X.iloc[:-weeks_to_predict].to_numpy()
        train_y = y.iloc[:-weeks_to_predict].to_numpy().reshape(-1,1)
        test_X = X.iloc[-weeks_to_predict].to_numpy().reshape(1,-1)
        test_y = np.asarray(y.iloc[-weeks_to_predict]).reshape(1,-1)
    else:
        train_X = X.iloc[:-weeks_to_predict].to_numpy()
        train_y = y.iloc[:-weeks_to_predict].to_numpy().reshape(-1,1)
        test_X = X.iloc[-weeks_to_predict:].to_numpy()
        test_y = np.asarray(y.iloc[-weeks_to_predict:]).reshape(-1,1)

    print(f'Shape of train_X: {train_X.shape}')
    print(f'Shape of train_y: {train_y.shape}')
    print(f'Shape of test_X: {test_X.shape}')
    print(f'Shape of test_y: {test_y.shape}')

    # scale data
    scaler = StandardScaler()
    train_X = scaler.fit_transform(train_X)
    test_X = scaler.transform(test_X)

    # instantiate model
    rf = RandomForestRegressor(max_depth=max_depth, random_state=seed_value)

    # fit the model
    rf.fit(train_X, train_y.ravel())

    # predict for train and test sets
    preds_test = rf.predict(test_X)
    preds_train = rf.predict(train_X)

    # plot
    plt.plot(data.index, data['Total Cases'].values)
    plt.plot(data.index[-weeks_to_predict:], preds_test, color='red')

    results = evaluate_model(test=test_y, predictions_test=preds_test, train=train_y, predictions_train=preds_train)



    results.insert(0, retrieve_name(rf))
    results.insert(1, df_name)
    results.insert(2, weeks_to_predict)

    results_matrix = pickle.load(open(r"..\data\results_matrix.pkl", "rb" ))

    results_matrix = pd.concat([results_matrix.T, pd.Series(results, index=results_matrix.columns)], axis=1).T

    pickle.dump(results_matrix, open(r"..\data\results_matrix.pkl", "wb" ))

    return results_matrix