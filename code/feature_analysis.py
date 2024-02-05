import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def feature_analysis(df):
    """
    Calculates the PnL properties for each strategy and compares it
    with benchmark strategies including long the asset and momentum.

    :param df: dataframe with observed and predicted asset returns
    :return: none
    """
    # create a momentum strategy from a rolling cumulative return
    # within the last 3 days
    df['yhat'] = df['y'].rolling(3).sum().shift(1)
    df['Mom'] = strategy_profit(df)

    # rename the observed returns to 'Long' strategy
    df = df.rename(columns={'y': 'Long'}).drop(columns='yhat')

    # strategy performance
    performance_metrics(df)

def performance_metrics(df):
    """
    Evaluates the performance of each strategy and shows the results.

    :param df: dataframe with strategy returns
    :return: dataframe with feature analysis statistics
    """
    # rearrange columns to have the proposed strategies first
    df = df[list(df.columns[1:]) + [df.columns[0]]]

    # Sharpe ratios and cumulative returns
    sharpe = df.mean() / df.std() * np.sqrt(252)
    cumret = df.cumsum()

    # estimate the drawdowns and the max drawdown MDD for every strategy
    dds = cumret - cumret.cummax()
    mdd = dds.min()

    # print messages and create plots
    print('\n\nStrategy performance\n', '\nSharpe Ratios\n', sharpe.to_string(),
          '\n\nMaximum Drawdowns\n', mdd.to_string(), '\n')
    plot(cumret, {'title': 'Strategy Cumulative Returns (Backtest)',
                  'ytitle': 'Cumulative Returns (Backtest)', 'filename': 'cumret'})
    plot(dds, {'title': 'Strategy Drawdowns (Backtest)',
               'ytitle': 'Drawdown', 'filename': 'drawdowns'})

def strategy_profit(df):
    """
    Calculates the strategy profit by assigning security weights based on
    a logistic function. The function is more conservative when the predicted
    return is near zero and more aggressive when the signal is strong.

    yhat is the todays'penny_freq prediction for next period. Therefore, when it'penny_freq
    transformed into a weight it must be lagged to be multiplied with tomorrow'penny_freq
    realized return. The sequence of operations is the following:
    i) b_{t-1} * f_t = yhat_t -> w_t (we buy w_t of security today)
    ii) realized profit next period: w_t * y_{t+1}

    :param df: dataframe with observed and predicted asset returns
    :return: series with the strategy'penny_freq realized profit
    """
    weights = 2 / (1 + np.exp(-10 * df['yhat'])) - 1
    return weights.shift(1) * df['y']

def plot(df, info):
    """
    Generates a time series plot.

    :param df: dataframe with strategy performance metrics
    :param info: plot configuration details
    :return: None
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    df = df.reset_index()
    df.plot(x_compat=True, kind='line', ax=ax, x='date', linewidth=2.3, rot=0)
    plt.rc('axes', linewidth=1)
    ax.set_xlim(left=df['date'].min(), right=df['date'].max())
    ax.set_title(info['title'], fontsize=18)
    ax.set_xlabel('Date', fontsize=15)
    ax.set_ylabel(info['ytitle'], fontsize=15)
    ax.legend(fontsize='x-large')
    # ax.xaxis.set_major_locator(mdates.YearLocator(1))
    # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    # ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.axhline(0, color='k', linestyle='-', linewidth=1)  # horizontal zero line
    ax.tick_params(which='major', direction='in', length=8, width=1.3, pad=5, labelsize=10)
    ax.tick_params(which='minor', direction='in', length=4, width=1, pad=5)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    fig.savefig('../output/{}.pdf'.format(info['filename']), bbox_inches='tight', format='pdf')
    print('Saved output/{}.pdf'.format(info['filename']))
