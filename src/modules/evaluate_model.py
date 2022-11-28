from sklearn.metrics import mean_squared_error, mean_absolute_error

class evaluate:
    """
    Calculate evaluation metrics for time series forecast
    """
    def __init__(self, train, test, predictions_train, predictions_test):
        self.train = train
        self.test = test
        self.predictions_train = predictions_train
        self.predictions_test = predictions_test


    def evaluate_model(self):
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
        
        if self.train != None:
            # Evaluate training set
            MSE_train = round(mean_squared_error(self.train, self.predictions_train),3)
            RMSE_train = round(mean_squared_error(self.train, self.predictions_train, squared=False),3)
            MAE_train = round(mean_absolute_error(self.train, self.predictions_train),3)

            print(f'Train MSE: {MSE_train}')
            print(f'Train RMSE: {RMSE_train}')
            print(f'Train MAE: {MAE_train}') 
        
        # Evaluate test set
        MSE_test = round(mean_squared_error(self.test, self.predictions_test),3)
        RMSE_test = round(mean_squared_error(self.test, self.predictions_test, squared=False),3)
        MAE_test = round(mean_absolute_error(self.test, self.predictions_test),3)

        print(f'Test MSE: {MSE_test}')
        print(f'Test RMSE: {RMSE_test}')
        print(f'Test MAE: {MAE_test}')