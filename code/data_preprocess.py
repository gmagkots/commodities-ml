import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform

def data_preprocess():
    """
    Data preprocessing before model estimation.

    :return: dataframe with processed feature and dependent variable time series
    """
    print('\nRaw data preprocessing')
    df = filter_data()
    df = cluster_data(df)
    df = first_differences(df)
    summary_statistics(df)
    return df

def filter_data():
    """
    Reads and transforms the raw time series data to mitigate scale differences.

    :return: dataframe with transformed feature and dependent variable data
    """
    # read the raw data and rename columns
    df = pd.read_excel('../input/Assessment_data.xlsx')
    df.columns = ['date', 'y'] + ['x' + str(col) if df[col].dtype == 'float' else
                                  'd' + str(col) for col in df.columns[2:]]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index(['date'])

    # remove integer/categorical variables with only zero values,
    # the severely imbalanced feature d60 (1500/1517 zeros), and
    # feature d53 because it makes a linear combination with d49-d52
    df = df.loc[:, (df != 0).any(axis=0)].drop(columns=['d53', 'd60'])

    # transform to ArcSinh each float column to mitigate scale discrepancies
    xcols = [col for col in df.columns if col.startswith('x')]
    df[xcols] = df[xcols].apply(np.arcsinh)

    return df

def cluster_data(df):
    """
    Clusters feature series by Pearson correlation and replaces them
    with representative median values to reduce dimensionality.

    :param df: dataframe with filtered input data
    :return: dataframe with clustered features and dependent variable
    """
    # estimate the correlation matrix of float columns
    cols = [col for col in df.columns if col.startswith('x')]
    corr = df.corr().loc[cols, cols]

    # perform scipy'penny_freq hierarchical clustering algo
    # distance metric: 1 - rho, rho is Pearson correlation
    distances, thr_corr = 1 - corr, 0.85
    linkage = sch.linkage(squareform(distances), method='complete')
    cl_ids = sch.fcluster(linkage, 1 - thr_corr, criterion='distance')

    # replace clustered features with the median time series
    clabel, cluster_dict = 0, {}
    for cl_id in range(min(cl_ids), max(cl_ids) + 1):
        cluster_members = [cols[i] for i, id in enumerate(cl_ids) if id == cl_id]
        if len(cluster_members) > 1:
            clabel += 1
            cluster_label = 'c' + str(clabel)
            df[cluster_label] = df[cluster_members].median(axis=1)
            df = df.drop(columns=cluster_members)
            cluster_dict[cluster_label] = cluster_members

    # export the cluster constituents in tabular format
    dft = pd.DataFrame.from_dict(cluster_dict, orient='index').T
    dft.to_csv('../output/clusters.csv', index=False)

    return df

def first_differences(df):
    """
    Takes first differences of float columns to diminish large
    AR(p) values and mitigate concerns about integrated processes.

    Exclude x1 that is constant for most dates. The long-term
    cyclicality persists for some features.

    :param df: dataframe with highly autocorrelated data
    :return: dataframe with first-differenced data
    """
    # autocorrelation threshold
    ar1_thr = 0.8

    # take first differences of float columns
    cols = [col for col in df.columns if col.startswith(('x', 'c'))]
    autocorrs = df[cols].apply(lambda x: x.autocorr())
    dcols = [col for col in cols if autocorrs[col] > ar1_thr and col != 'x1']
    df[dcols] = df[dcols].diff()
    df = df.dropna()

    return df

def summary_statistics(df):
    """
    Provides summary statistics of the processed input data.

    :param df: dataframe with processed data
    :return: None
    """
    # summary statistics of numerical features
    prc = [0.01, 0.25, 0.5, 0.75, 0.99]
    dft = df.select_dtypes(include='float')
    dfs = [dft.describe(percentiles=prc), pd.DataFrame(dft.skew()).T.rename(index={0: 'skewness'}),
           pd.DataFrame(dft.kurt()).T.rename(index={0: 'kurtosis'}),
           pd.DataFrame(dft.apply(lambda x: x.autocorr())).T.rename(index={0: 'AR(1)'})]
    stat_float = pd.concat(dfs).T.rename_axis('Variable').reset_index().drop(columns='count')
    stat_float.columns = [s if s.startswith('AR') else s.capitalize() for s in stat_float.columns]
    print('\nFull-sample summary statistics (numerical variables).')
    print(stat_float)
    stat_float.to_csv('../output/summary_stat_numerical.csv', index=False)

    # summary statistics of categorical features
    dft = df.select_dtypes(include='int64').applymap(str)
    stat_int = dft.describe()
    print('\nFull-sample summary statistics (categorical variables).')
    print(stat_int)
    stat_int.to_csv('../output/summary_stat_categorical.csv', index=False)
