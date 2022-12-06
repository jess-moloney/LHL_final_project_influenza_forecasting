import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
sns.set_context(context='poster')
plt.rcParams['figure.figsize'] = [20,10]

def plot_target_relationship(data, feature, feature_title, target_title, color):
    """
            Visualize the relationship between a target variable and a feature using a barplot and a strip plot.
            
            Args:
                data (dataframe): dataframe including target and feature
                feature (str): column name of feature
                feature_title (str): name of feature to include in plot title
                target_title (str): name of target to include in plot title
                color (str): color to use in the bar plot (refer to https://matplotlib.org/stable/gallery/color/named_colors.html)
                
            Returns:
                printout of shape of training X and y, shape of test X and y, plot of forecast, result_matrix with MSE, RMSE, MAE for train and test sets appended

        """ 
    fig, axes = plt.subplots(1,2, sharex=False)
    fig.suptitle(f'Mean {target_title} by {feature_title}', y=.98, va='bottom')

    sns.barplot(ax=axes[0], data=data, x=feature, y='y1', color=color, saturation=0.3)
    sns.stripplot(ax=axes[1], data=data,x=feature, y='y1', alpha = 0.7)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].set_ylabel(target_title)
    axes[0].set_xlabel(feature_title)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].set_ylabel('')
    axes[1].set_xlabel(feature_title)
