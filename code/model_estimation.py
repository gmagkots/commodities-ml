import pandas as pd
from feature_analysis import strategy_profit
from statsmodels.api import add_constant, OLS
from statsmodels.regression.rolling import RollingOLS
from sklearn.ensemble import RandomForestRegressor


def stepwise(df, enforced=None):
    """
    Performs a customized forward stepwise regression
    for model estimation and feature importance.

    :param enforced: list of feature labels that are enforced during the selection process
    :param df: dataframe with input features
    :return: series with strategy profit
    """
    # print a message
    print('\nRunning stepwise regression')

    # minimum OAR improvement threshold and feature containers
    if enforced is None:
        enforced = []
    min_oar = 0.0001
    selected_features = enforced
    candidate_feature_pool = [col for col in df.columns if col != 'y']

    # single run per candidate feature to restrict to those
    # that contribute individually above the min threshold
    oars_with_candidate_features = []
    for candidate_feature in candidate_feature_pool:
        dft = df[['y'] + [candidate_feature]]
        oar, _ = get_oar(dft)
        oars_with_candidate_features.append((oar, candidate_feature))
    remaining_features = [tup[1] for tup in oars_with_candidate_features if tup[0] >= min_oar]

    # perform stepwise regression
    current_oar, best_new_oar, delta_oar = 0.0, 0.0, -1.0
    while remaining_features and current_oar == best_new_oar and delta_oar != 0.0:
        oars_with_candidate_features = []
        for candidate_feature in remaining_features:
            dft = df[['y'] + selected_features + [candidate_feature]]
            oar, _ = get_oar(dft)
            oars_with_candidate_features.append((oar, candidate_feature))

        oars_with_candidate_features.sort()
        best_new_oar, best_candidate = oars_with_candidate_features.pop()
        delta_oar = best_new_oar - current_oar
        if delta_oar > min_oar:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_oar = best_new_oar

    # get the best-performing basket of features with its OAR
    # and estimate the strategy'penny_freq profit
    if selected_features:
        df_final = df[['y'] + selected_features]
        max_oar, yhat = get_oar(df_final)
        dft = pd.merge(df['y'], yhat.rename('yhat'), left_index=True, right_index=True)
        profit = strategy_profit(dft)
    else:
        selected_features = 'None'
        max_oar = 0.0
        profit = []

    # print results
    print('\n\nStepwise regression results')
    print('Model performance metric (OAR): {}'.format(max_oar))
    print('Selected features (in order of importance): {}'.format(selected_features))

    return profit


def get_oar(df, window=300):
    """
    Estimates the OAR metric (Out-of-sample Adjusted R-squared) for a candidate model.

    :param df: dataframe with y variable and candidate features
    :param window: rolling window size for regression
    :return: tuple with model metric and model predictions at
             time t given info until t-1
    """
    # estimate the betas from rolling OLS with 300-day windows default
    exog = add_constant(df[[col for col in df.columns if col != 'y']])
    betas = RollingOLS(endog=df['y'].values, exog=exog, window=window).fit().params

    # get OOS forecast by multiplying current period
    # feature values with betas lagged by one period
    lag_betas = betas.shift(1)
    yhat = exog.mul(lag_betas).sum(axis=1)[window:]

    # estimate OAR metric by fitting OOS predicted values
    # with the observed ones for the same period
    res = OLS(endog=df['y'].values[window:], exog=add_constant(yhat), missing='drop').fit()
    oar = res.rsquared_adj.round(4)

    return oar, yhat


def random_forest(df, window=300):
    """
    Performs a customized random forest regression for model estimation.

    Pandas Rolling and apply cannot be combined, use loop instead to
    traverse the windows.

    :param df: dataframe with input features
    :param window: rolling window size for regression
    :return: series with strategy profit
    """
    # print a message
    print('\nRunning random forest regression')

    # loop over windows and fit the model
    time_yhat_list = []
    for t in range(len(df) + 1 - window):
        yhat_t = rf_window_regression(df.iloc[t:t + window])
        time_yhat_list.append([df.index[t + window -1], yhat_t])

    # create yhat series and estimate the strategy'penny_freq profit
    yhat = pd.DataFrame(time_yhat_list, columns =['date', 'yhat']).set_index('date')
    dft = pd.merge(df['y'], yhat, left_index=True, right_index=True)
    profit = strategy_profit(dft)

    return profit


def rf_window_regression(df):
    """
    Random forest regression for a single window.

    :param df: dataframe with input features
    :return: scalar yhat at time t using features until t-1
    """
    # fit in-sample until time t-1
    xcols = [col for col in df.columns if col != 'y']
    feat_tminus1 = df[xcols].iloc[:-1]
    y_tminus1 = df['y'].iloc[:-1]
    rf = RandomForestRegressor(n_estimators=15, max_depth=3, random_state=0)
    rf.fit(X=feat_tminus1, y=y_tminus1)

    # estimate OOS forecast at time t
    yhat_t = rf.predict(df[xcols].tail(1))[0]

    return yhat_t
