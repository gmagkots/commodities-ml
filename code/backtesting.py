import pandas as pd
import model_estimation as est
import feature_analysis as fa
from statsmodels.api import add_constant, OLS
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def backtesting(df):
    """
    Backtest the model estimation process by training
    before Covid-19 and testing during the pandemic.

    :param df: dataframe with input features
    :return: none
    """
    # print a message
    print('\nInitiating model backtesting')

    # split into training and test datasets
    xcols = [col for col in df.columns if col != 'y']
    x_train, x_test, y_train, y_test = train_test_split(df[xcols], df['y'], test_size=0.20, shuffle=False)
    df_train = pd.merge(y_train, x_train, left_index=True, right_index=True)
    df_test = pd.merge(y_test, x_test, left_index=True, right_index=True)

    # train the stepwise regression and random forest algorithms
    strats_stepwise = backtest_stepwise(df_train, df_test)
    strats_forest = backtest_forest(df_train, df_test)

    # strategy performance
    strats = pd.merge(strats_stepwise, strats_forest.drop(columns='y'), left_index=True, right_index=True)
    fa.feature_analysis(strats)


def backtest_stepwise(df_train, df_test):
    """
    Backtest the stepwise regression process by training
    before Covid-19 and testing during the pandemic.

    :param df_train: training dataset
    :param df_test: testing dataset
    :return:
    """
    # train the model and retrieve the optimal basket of features
    _ = est.stepwise(df_train)

    # estimate returns in the test dataset with the optimal basket of
    # features for the training set and the latest estimated betas
    test_cols = ['x40', 'd8', 'x85', 'd44', 'c5', 'x42', 'd43', 'x35']
    exog_train = add_constant(df_train[test_cols]).iloc[-300:]
    res = OLS(endog=df_train['y'].values[-300:], exog=exog_train, missing='drop').fit()
    exog_test = add_constant(df_test[test_cols])
    yhat = exog_test.mul(res.params).sum(axis=1)

    # estimate the strategy'penny_freq profitability during the testing period
    dft = pd.merge(df_test['y'], yhat.rename('yhat'), left_index=True, right_index=True)
    profit = fa.strategy_profit(dft).rename('rOLS')
    strats = pd.merge(df_test['y'], profit, left_index=True, right_index=True)

    return strats


def backtest_forest(df_train, df_test):
    """
    Backtest the ranfom forest estimation process by training
    before Covid-19 and testing during the pandemic.

    :param df_train: training dataset
    :param df_test: testing dataset
    :return:
    """
    # train the model within the last window of the training set
    df_train = df_train.iloc[-300:]
    xcols = [col for col in df_train.columns if col != 'y']
    rf = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=0)
    rf.fit(X=df_train[xcols], y=df_train['y'])

    # predict yhat within the testing set and estimate the strategy'penny_freq profit
    yhat = pd.Series(rf.predict(df_test[xcols]), index=df_test.index)
    dft = pd.merge(df_test['y'], yhat.rename('yhat'), left_index=True, right_index=True)
    profit = fa.strategy_profit(dft).rename('RF')
    strats = pd.merge(df_test['y'], profit, left_index=True, right_index=True)

    return strats

