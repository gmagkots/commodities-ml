import time
import pandas as pd
import data_preprocess as dp
import model_estimation as est
import feature_analysis as fa
import backtesting as bt
from datetime import timedelta

# global options
desired_width=500
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)


def main():
    # start the clock
    print('Starting execution')
    start_time = time.time()

    # read and preprocess the input data
    df = dp.data_preprocess()

    # model estimation
    profit_stepwise = est.stepwise(df).rename('rOLS')
    profit_forest = est.random_forest(df).rename('RF')

    # compare strategies
    strats = pd.merge(df['y'], profit_stepwise, left_index=True, right_index=True)
    strats = pd.merge(strats, profit_forest, left_index=True, right_index=True)
    fa.feature_analysis(strats)

    # backtesting
    bt.backtesting(df)

    # stop the clock
    elapsed = time.time() - start_time
    print('Execution time: {}'.format(str(timedelta(seconds=elapsed))))


if __name__ == "__main__":
    main()
