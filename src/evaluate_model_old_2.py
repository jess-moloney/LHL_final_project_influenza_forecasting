from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(test, predictions_test, train=None, predictions_train=None):
    """
    Calculate MSE_train, RMSE_train, MAE_train, and MSE_test, RMSE_test, MAE_test to evaluate time series forecast

    MSE - weights large errors more than smaller ones, good for penalizing large errors, loses unit because squared
    RMSE - more interpretable than MSE because avoids losing units, also penalizes large errors
    MAE - doesn't penalize large errors as much

    Args:
        train (list of int, optional): values of the training set
        test (list of int): values of the test set
        predictions_train (list of int, optional): predicted values for the training set
        predictions_test (list of int, optional): predicted values for the test set

    Returns:
        print out of training and test scores

   """ 
    results = []

    if train.any():
        # Evaluate training set
        MSE_train = round(mean_squared_error(train, predictions_train),3)
        RMSE_train = round(mean_squared_error(train, predictions_train, squared=False),3)
        MAE_train = round(mean_absolute_error(train, predictions_train),3)

        results.append(MSE_train)
        results.append(RMSE_train)
        results.append(MAE_train)
    
    # Evaluate test set
    MSE_test = round(mean_squared_error(test, predictions_test),3)
    RMSE_test = round(mean_squared_error(test, predictions_test, squared=False),3)
    MAE_test = round(mean_absolute_error(test, predictions_test),3)

    results.append(MSE_test)
    results.append(RMSE_test)
    results.append(MAE_test)

    return results