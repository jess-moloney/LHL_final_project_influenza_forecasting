# Modeling
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor

import pandas as pd

# Visualization

# Plotly
import plotly.graph_objs as go
from plotly.offline import iplot, plot, init_notebook_mode
init_notebook_mode(connected=True)
import plotly_express as px

# cufflinks is a wrapper on plotly
import cufflinks as cf
cf.go_offline(connected=True)

def plot_intervals(predictions, mid=False, start=None, stop=None, title=None):
        """
        https://github.com/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb
        Function for plotting prediction intervals as filled area chart.
        
        :param predictions: dataframe of predictions with lower, upper, and actual columns (named for the target)
        :param whether to show the mid prediction
        :param start: optional parameter for subsetting start of predictions
        :param stop: optional parameter for subsetting end of predictions
        :param title: optional string title
        
        :return fig: plotly figure
        """
        # Subset if required
        predictions = (
            predictions.loc[start:stop].copy()
            if start is not None or stop is not None
            else predictions.copy()
        )
        data = []

        # Lower trace will fill to the upper trace
        trace_low = go.Scatter(
            x=predictions.index,
            y=predictions["lower"],
            fill="tonexty",
            line=dict(color="darkblue"),
            fillcolor="rgba(173, 216, 230, 0.4)",
            showlegend=True,
            name="lower",
        )
        # Upper trace has no fill
        trace_high = go.Scatter(
            x=predictions.index,
            y=predictions["upper"],
            fill=None,
            line=dict(color="orange"),
            showlegend=True,
            name="upper",
        )
        # Must append high trace first so low trace fills to the high trace
        data.append(trace_high)
        data.append(trace_low)
        
        if mid:
            trace_mid = go.Scatter(
            x=predictions.index,
            y=predictions["mid"],
            fill=None,
            line=dict(color="green"),
            showlegend=True,
            name="mid",
        )
            data.append(trace_mid)

        # Trace of actual values
        trace_actual = go.Scatter(
            x=predictions.index,
            y=predictions["actual"],
            fill=None,
            line=dict(color="black"),
            showlegend=True,
            name="actual",
        )
        data.append(trace_actual)

        # Layout with some customization
        layout = go.Layout(
            height=600,
            width=1200,
            title=dict(text="Prediction Intervals" if title is None else title),
            yaxis=dict(title=dict(text="Cases")),
            xaxis=dict(
                rangeselector=dict(
                    buttons=list(
                        [
                            dict(count=1, label="1d", step="day", stepmode="backward"),
                            dict(count=7, label="1w", step="day", stepmode="backward"),
                            dict(count=1, label="1m", step="month", stepmode="backward"),
                            dict(count=1, label="YTD", step="year", stepmode="todate"),
                            dict(count=1, label="1y", step="year", stepmode="backward"),
                            dict(step="all"),
                        ]
                    )
                    ),
                rangeslider=dict(visible=True),
                type="date",
            ),
        )

        fig = go.Figure(data=data, layout=layout)

        # Make sure font is readable
        fig["layout"]["font"] = dict(size=15)
        fig.layout.template = "plotly_white"
        return fig

def calculate_error(predictions):
    """
    https://github.com/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb
    Calculate the absolute error associated with prediction intervals
    
    :param predictions: dataframe of predictions
    :return: None, modifies the prediction dataframe
    
    """
    predictions['absolute_error_lower'] = (predictions['lower'] - predictions["actual"]).abs()
    predictions['absolute_error_upper'] = (predictions['upper'] - predictions["actual"]).abs()
    
    predictions['absolute_error_interval'] = (predictions['absolute_error_lower'] + predictions['absolute_error_upper']) / 2
    predictions['absolute_error_mid'] = (predictions['mid'] - predictions["actual"]).abs()
    
    predictions['in_bounds'] = predictions["actual"].between(left=predictions['lower'], right=predictions['upper'])

def show_metrics(metrics):
    """
    Make a boxplot of the metrics associated with prediction intervals
    
    :param metrics: dataframe of metrics produced from calculate error 
    :return fig: plotly figure
    """
    percent_in_bounds = metrics['in_bounds'].mean() * 100
    metrics_to_plot = metrics[[c for c in metrics if 'absolute_error' in c]]

    # Rename the columns
    metrics_to_plot.columns = [column.split('_')[-1].title() for column in metrics_to_plot]

    # Create a boxplot of the metrics
    fig = px.box(
        metrics_to_plot.melt(var_name="metric", value_name='Absolute Error'),
        x="metric",
        y="Absolute Error",
        color='metric',
        title=f"Error Metrics Boxplots    In Bounds = {percent_in_bounds:.2f}%",
        height=800,
        width=1000,
        points=False,
    )

    # Create new data with no legends
    d = []

    for trace in fig.data:
        # Remove legend for each trace
        trace['showlegend'] = False
        d.append(trace)

    # Make the plot look a little better
    fig.data = d
    fig['layout']['font'] = dict(size=20)
    return fig

class GradientBoostingPredictionIntervals(BaseEstimator):
    """
    https://github.com/WillKoehrsen/Data-Analysis/blob/master/prediction-intervals/prediction_intervals.ipynb
    Model that produces prediction intervals with a Scikit-Learn inteface
    
    :param lower_alpha: lower quantile for prediction, default=0.1
    :param upper_alpha: upper quantile for prediction, default=0.9
    :param **kwargs: additional keyword arguments for creating a GradientBoostingRegressor model
    """

    def __init__(self, lower_alpha=0.1, upper_alpha=0.9, **kwargs):
        self.lower_alpha = lower_alpha
        self.upper_alpha = upper_alpha

        # Three separate models
        self.lower_model = GradientBoostingRegressor(
            loss="quantile", alpha=self.lower_alpha, **kwargs
        )
        self.mid_model = GradientBoostingRegressor(loss="ls", **kwargs)
        self.upper_model = GradientBoostingRegressor(
            loss="quantile", alpha=self.upper_alpha, **kwargs
        )
        self.predictions = None

    def fit(self, X, y):
        """
        Fit all three models
            
        :param X: train features
        :param y: train targets
        
        TODO: parallelize this code across processors
        """
        self.lower_model.fit(X, y)
        self.mid_model.fit(X, y)
        self.upper_model.fit(X, y)

    def predict(self, X, y):
        """
        Predict with all 3 models 
        
        :param X: test features
        :param y: test targets
        :return predictions: dataframe of predictions
        
        TODO: parallelize this code across processors
        """
        predictions = pd.DataFrame(y)
        predictions["lower"] = self.lower_model.predict(X)
        predictions["mid"] = self.mid_model.predict(X)
        predictions["upper"] = self.upper_model.predict(X)
        self.predictions = predictions

        return predictions

    def plot_intervals(self, mid=False, start=None, stop=None):
        """
        Plot the prediction intervals
        
        :param mid: boolean for whether to show the mid prediction
        :param start: optional parameter for subsetting start of predictions
        :param stop: optional parameter for subsetting end of predictions
    
        :return fig: plotly figure
        """

        if self.predictions is None:
            raise ValueError("This model has not yet made predictions.")
            return
        
        fig = plot_intervals(self.predictions, mid=mid, start=start, stop=stop)
        return fig

    def calculate_and_show_errors(self):
        """
        Calculate and display the errors associated with a set of prediction intervals
        
        :return fig: plotly boxplot of absolute error metrics
        """
        if self.predictions is None:
            raise ValueError("This model has not yet made predictions.")
            return
        
        calculate_error(self.predictions)
        fig = show_metrics(self.predictions)
        return fig

    