import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_predict
import warnings


def arima_model(data, weeks_to_predict, p, d, q):
    """
        Split data, the train, fit, evaluate and plot an ARIMA model for a specified number of weeks ahead to forecast
        
        Args:
            data (dataframe with DatetimeIndex): time series to train and validate model on
            weeks_to_predict (int): number of weeks ahead to predict
            p (int): the time lag after which partial autocorrelation is no longer significant (aka no longer has an effect on current value). Choose using partial autocorrelation plot.
            d (int): the number of times differencing needs to be applied to achieve stationarity in the time series (aka pass the ADF test)
            q (int): the time lag after which autocorrelation is no longer significant. Choose using autocorrelation plot.
            
        Returns:
            Length of training set, length of test set, MSE, RMSE, MAE, plot of forecast

    """ 
    warnings.filterwarnings("ignore")
    
    # Split to train and test
    X = data
    train, test = X.iloc[:-weeks_to_predict], X.iloc[-weeks_to_predict:]
    print(f'Length of training set: {len(train)}')
    print(f'Length of test set: {len(test)}')

    # Instantiate model
    model = ARIMA(train, order=(p,d,q), freq='W-SAT')

    # Fit model
    fitted_model = model.fit()
    print(f'Fitted model summary: \n{fitted_model.summary()}')

    # Forecast
    forecast = fitted_model.forecast(steps=weeks_to_predict)
    print(f'Forecast: \n{forecast}')

    # Plot forecast
    # fig, ax = plt.subplots()
    # forecast_start = str(data.iloc[-weeks_to_predict].name)
    # plot_start = str(data.iloc[-52].name)
    # stop = str(data.iloc[-1].name)
    # ax = data[plot_start:stop].plot(ax=ax)
    # plot_predict(fitted_model, forecast_start, stop, ax=ax)
    # plt.show()

    return test, forecast