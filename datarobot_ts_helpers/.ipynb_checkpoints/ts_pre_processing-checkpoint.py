import numpy as np


#####################
# Preprocessing Funcs
#####################


def dataset_reduce_memory(df):
    """
    Recast numerics to lower precision
    """
    for c in df.select_dtypes(include=['float64']).columns:
        df[c] = df[c].astype(np.float32)
    for c in df.select_dtypes(include=['int64']).columns:
        df[c] = df[c].astype(np.int32)
    return df


def create_series_id(df, cols_to_concat, convert=True):
    """
    Concatenate columns

    Returns:
    --------
    pandas Series
    """
    df = df[cols_to_concat].copy()
    non_strings = [c for c in df[cols_to_concat] if df[c].dtype != 'object']

    if len(non_strings) > 0:
        if convert:
            df[non_strings] = df[non_strings].applymap(str)
        else:
            raise TypeError("columns must all be type str")

    df['series_id'] = df[cols_to_concat].apply(lambda x: '_'.join(x), axis=1)
    return df['series_id']


def _create_cross_series_feature(df, group, col, func):
    """
    Creates aggregate functions for statistics within a cluster
    df: pandas df
    group: str
        Column name used for groupby
    col: str
        Column name on which functions should be applied
    func: list
        list of pandas-compatible .transform(func) of aggregation functions
        
    Returns:
    --------
    pandas df
    """
    
    col_name = col + '_' + func
    df.loc[:, col_name] = df.groupby(group)[col].transform(func)
    return df


def create_cross_series_features(df, group, cols, funcs):
    """
    Create custom aggregations across groups
    
    df: pandas df
    group: str
        Column name used for groupby
    col: str
        Column name on which functions should be applied
    func: list
        list of pandas-compatible .transform(func) of aggregation functions

    Returns:
    --------
    pandas df with new cross series features

    Example:
    --------
    df_agg = create_cross_series_features(df,
                                          group=[date_col,'Cluster'],
                                          cols=[target,'feat_1'],
                                          funcs=['mean','std'])
    """
    for c in cols:
        for f in funcs:
            df = _create_cross_series_feature(df, group, c, f)
    return df.reset_index(drop=True)


def get_zero_inflated_series(df, ts_settings, cutoff=0.99):
    """
    Identify series where the target is 0.0 in more than x% of the rows

    df: pandas df
    ts_settings: dict
        Parameters of datetime DR project
    cutoff: np.float
        Threshold for removal of zero-inflated series. Retained series must be present in row >= cutoff

    Returns:
    --------
    List of series
    """
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    target = ts_settings['target']

    df = df.groupby([series_id])[target].apply(lambda x: (x.dropna() == 0).mean())
    series = df[df >= cutoff].index.values

    return series


def drop_zero_inflated_series(df, ts_settings, cutoff=0.99):
    """
    Remove series where the target is 0.0 in more than x% of the rows

    df: pandas df
    ts_settings: dict
        Parameters of datetime DR project
    cutoff: np.float
        Threshold for removal of zero-inflated series. Retained series must be present in row >= cutoff

    Returns:
    --------
    pandas df
    """
    
    series_id = ts_settings['series_id']

    series_to_drop = get_zero_inflated_series(df, ts_settings, cutoff=cutoff)

    if len(series_to_drop) > 0:
        print('Dropping ', len(series_to_drop), ' zero-inflated series')
        df = df.loc[~df[series_id].isin(series_to_drop), :].reset_index(drop=True)
        print('Remaining series: ', len(df[series_id].unique()))
    else:
        print('There are no zero-inflated series to drop')

    return df


def sample_series(df, series_id, date_col, target, x=1, method='random', **kwargs):
    """
    Sample series

    x: percent of series to sample
    random: sample x% of the series at random
    target: sample the largest x% of series
    timespan: sample the top x% of series with the longest histories

    """
    if (x > 1) | (x < 0):
        raise ValueError('x must be between 0 and 1')

    df.sort_values(by=[date_col, series_id], ascending=True, inplace=True)
    series = round(x * len(df[series_id].unique()))

    if method == 'random':
        series_to_keep = np.random.choice(df[series_id].values, size=series)

    elif method == 'target':
        series_to_keep = (
            df.groupby([series_id])[target]
            .mean()
            .sort_values(ascending=False)
            .reset_index()
            .loc[0:series, series_id]
        )

    elif method == 'timespan':
        max_timespan = df[date_col].max() - df[date_col].min()
        series_timespans = (
            df.groupby([series_id])[date_col]
            .apply(lambda x: x.max() - x.min())
            .sort_values(ascending=False)
            .reset_index()
        )
        series_to_keep = series_timespans.loc[0:series, series_id]
        if kwargs.get('full_timespan'):
            series_to_keep = series_timespans.loc[series_timespans == max_timespan, series_id]

    else:
        raise ValueError('Method not supported. Must be either random, target, or timespan')

    sampled_df = df.loc[df[series_id].isin(series_to_keep), :]

    return sampled_df.reset_index(drop=True)


def drop_series_w_gaps(df, series_id, date_col, target, max_gap=1, output_dropped_series=False):
    """
    Removes series with missing rows
    
    df: pandas df
    series_id: str
        Column name with series identifier
    date_col: str
        Column name of datetime column
    target: str
        Column name of target column
    max_gap: int
        number of allowed missing timestep
    output_dropped_series: bool (optional)
        allows return of pandas df of series that do not satisfy max_gap criteria
    
    Returns:
    --------
    pandas df(s)
    """
    
    if not isinstance(max_gap, int):
        raise TypeError('max gap must be an int')

    df.sort_values(by=[date_col, series_id], ascending=True, inplace=True)
    series_max_gap = df.groupby([series_id]).apply(lambda x: x[date_col].diff().max())
    median_timestep = df.groupby([series_id])[date_col].diff().median()
    series_to_keep = series_max_gap[(series_max_gap / median_timestep) <= max_gap].index.values

    sampled_df = df.loc[df[series_id].isin(series_to_keep), :]
    dropped_df = df.loc[~df[series_id].isin(series_to_keep), :]

    if output_dropped_series:
        return sampled_df, dropped_df
    else:
        return sampled_df
