from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import pandas as pd
import os.path

def evaluate_model(train, test, preds_train, preds_test, dataset, forecast_weeks, model):
  
    # evaluate
    MSE_train = round(mean_squared_error(train, preds_train),2)
    MSE_test = round(mean_squared_error(test, preds_test),2)
    RMSE_train = round(mean_squared_error(train, preds_train, squared=False),2)
    RMSE_test = round(mean_squared_error(test, preds_test, squared=False),2)
    MAE_train = round(mean_absolute_error(train, preds_train),2)
    MAE_test = round(mean_absolute_error(test, preds_test),2)
    R2_train = r2_score(train, preds_train)
    R2_test = r2_score(test, preds_test)

    # MSE_train = round(mean_squared_error(train.iloc[:,forecast_weeks-1], preds_train[:,forecast_weeks-1]),2)
    # MSE_test = round(mean_squared_error(test.iloc[:,forecast_weeks-1], preds_test[:,forecast_weeks-1]),2)
    # RMSE_train = round(mean_squared_error(train.iloc[:,forecast_weeks-1], preds_train[:,forecast_weeks-1], squared=False),2)
    # RMSE_test = round(mean_squared_error(test.iloc[:,forecast_weeks-1], preds_test[:,forecast_weeks-1], squared=False),2)
    # MAE_train = round(mean_absolute_error(train.iloc[:,forecast_weeks-1], preds_train[:,forecast_weeks-1]),2)
    # MAE_test = round(mean_absolute_error(test.iloc[:,forecast_weeks-1], preds_test[:,forecast_weeks-1]),2)
    # R2_train = r2_score(train.iloc[:,forecast_weeks-1], preds_train[:,forecast_weeks-1])
    # R2_test = r2_score(test.iloc[:,forecast_weeks-1], preds_test[:,forecast_weeks-1])

    results = []

    results.append(MSE_train)
    results.append(MSE_test)
    results.append(RMSE_train)
    results.append(RMSE_test)
    results.append(MAE_train)
    results.append(MAE_test)
    results.append(R2_train)
    results.append(R2_test)
   
    results.insert(0, model)
    results.insert(1, dataset)
    results.insert(2, forecast_weeks)
    # results.insert(-1, train.columns)

    file_name = r"..\data\results_matrix_" + model + ".pkl"

    if os.path.exists(file_name) == False:

        results_matrix = pd.DataFrame(columns=['Model','Dataset','Weeks-ahead Forecast','MSE_train','MSE_test','RMSE_train','RMSE_test','MAE_train','MAE_test','R2_train','R2_test'])
        pickle.dump(results_matrix, open(file_name, "wb" ))

    results_matrix = pickle.load(open(file_name, "rb" ))
   
    results_matrix = pd.concat([results_matrix.T, pd.Series(results, index=results_matrix.columns)], axis=1).T

    pickle.dump(results_matrix, open(file_name, "wb" ))

    return results_matrix