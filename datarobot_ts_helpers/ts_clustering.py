import datetime as dt
import operator
import datarobot as dr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from statsmodels.tsa.stattools import pacf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import umap

from .ts_metrics import *
from .ts_modeling import create_dr_project
from .ts_projects import get_preds_and_actuals


####################
# Series Clustering
####################


def _split_series(df, series_id, target, by='quantiles', cuts=5, split_col='Cluster'):
    """
    Split series into clusters by rank or quantile  of average target value

    by: str
        Rank or quantiles
    cuts: int
        Number of clusters
    split_col: str
        Name of new column

    Returns:
    --------
    pandas df
    """
    group = df.groupby([series_id]).mean()

    if by == 'quantiles':
        group[split_col] = pd.qcut(group[target], cuts, labels=np.arange(1, cuts + 1))
    elif by == 'rank':
        group[split_col] = pd.cut(group[target], cuts, labels=np.arange(1, cuts + 1))
    else:
        raise ValueError(f'{by} is not a supported value. Must be set to either quantiles or rank')

    df = df.merge(
        group[split_col], how='left', left_on=series_id, right_index=True, validate='many_to_one'
    )

    df[split_col] = df[split_col].astype('str')
    n_clusters = len(df[split_col].unique())
    mapper_clusters = {k: v for (k, v) in zip(df[split_col].unique(), range(1, n_clusters + 1))}
    df[split_col] = df[split_col].map(mapper_clusters)

    return df.reset_index(drop=True)


def _get_pacf_coefs(df, col, nlags, alpha, scale, scale_method):
    """
    Helper function for add_cluster_labels()

    df: pandas df
    col: str
        Series name
    nlags: int
        Number of AR coefficients to include in pacf
    alpha: float
        Cutoff value for p-values to determine statistical significance
    scale: boolean
        Whether to standardize input data
    scale_method: str
        Choose from 'min_max' or 'normalize'

    Returns:
    --------
    List of AR(n) coefficients

    """
    if scale:
        if scale_method == 'min_max':
            df = df.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), axis=0)
        elif scale_method == 'normalize':
            df = df.apply(lambda x: (x - np.mean(x)) / np.std(x), axis=0)
        else:
            raise ValueError(
                f'{scale_method} is not a supported value. scale_method must be set to either min_max or normalize'
            )

    # if df[col].dropna().shape[0] == 0:
    #     print(col, df[col].dropna())
#     print('Running PAC...')
    clf = pacf(df[col].dropna(), method='ols', nlags=nlags, alpha=alpha)
    if alpha:
        coefs = clf[0][1:]
        zero_in_interval = [not i[0] < 0 < i[1] for i in clf[1][1:]]
        adj_coefs = [c if z else 0.0 for c, z in zip(coefs, zero_in_interval)]
        return adj_coefs
    else:
        coefs = clf[1:]
        return coefs


def _get_performance_cluster_results(df, ts_settings, n_clusters, max_clusters):
    """
    Helper function for add_cluster_labels()

    Use series acccuracy from an XGBoost model to cluster series

    Returns:
    --------
    distance matrix

    """

    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    target = ts_settings['target']

    project_name = 'Temp'

    settings = ts_settings.copy()
    settings['mode'] = 'manual'
    settings['number_of_backtests'] = 1

    if settings['max_date'] is None:
        settings['max_date'] = df[date_col].max() - dt.timedelta(days=90)
    else:
        settings['max_date'] = pd.to_datetime(settings['max_date']) - dt.timedelta(days=90)

    temp = create_dr_project(df, project_name, settings)

    blueprints = temp.get_blueprints()
    bp = [
        bp
        for bp in blueprints
        if 'eXtreme Gradient Boosted Trees' in bp.model_type
        if 'Decay' not in bp.model_type
        if 'Multiseries ID' not in ','.join(bp.processes)
    ][0]
    featurelist = [
        fl for fl in temp.get_modeling_featurelists() if '(average baseline)' in fl.name
    ][
        0
    ]  # not sure if there's a better default to use here
    duration = dr.DatetimePartitioning.get(temp.id).primary_training_duration
    print('Training duration: ', duration)
    
    model_job = temp.train_datetime(
        blueprint_id=bp.id, featurelist_id=featurelist.id, training_duration=duration
    )
    print(f'Training {bp.model_type} Model')
    model_job.wait_for_completion(max_wait=10000)

    temp.unlock_holdout()

    df = get_preds_and_actuals(df, [temp], settings, n_models=1, data_subset='holdout')

    df['residual'] = np.abs(df['prediction'] - df[target])
    df = df[[series_id, date_col, 'residual', 'forecast_distance']].copy()
    df = df.groupby([series_id, date_col]).agg({'residual': 'mean'}).reset_index()
    df = df.pivot(index=date_col, columns=series_id, values='residual')
    df = df.fillna(df.mean()).corr(method='pearson')

    dist_df = df.apply(lambda x: x.fillna(x.mean()), axis=1)

    temp.delete()

    return dist_df


def _get_optimal_n_clusters(df, n_series, max_clusters, plot=True):
    """
    Helper function for add_cluster_labels()

    Get the number of clusters that results in the max silhouette score

    Returns:
    --------
    int

    """
    clusters = list(np.arange(min(max_clusters, n_series)) + 2)[:-1]
    print(f'Testing {clusters[0]} to {clusters[-1]} clusters')
    scores = {}
    d = []
    for c in clusters:
        kmean = KMeans(n_clusters=c).fit(df)
        d.append(sum(np.min(cdist(df, kmean.cluster_centers_, 'euclidean'), axis=1)) / df.shape[0])
        preds = kmean.predict(df)
        score = silhouette_score(df, preds, metric='euclidean')
        scores[c] = score
        print(f'For n_clusters = {c}, silhouette score is {score}')

    n_clusters = max(scores.items(), key=operator.itemgetter(1))[0]
    best_score = scores[n_clusters]
    print(f'optimal n_clusters = {n_clusters}, max silhouette score is {best_score}')

    if max_clusters > 2:
        if plot:
            fig = px.line(x=clusters, y=d)
            fig.update_layout(height=500, width=750, title_text='Kmeans Optimal Number of Clusters')
            fig.update_xaxes(title='Number of Clusters', range=[clusters[0], clusters[-1]])
            fig.update_yaxes(title='Distortion')
            fig.show()

    return n_clusters


def add_cluster_labels(
    df,
    ts_settings,
    method,
    nlags=None,
    scale=True,
    scale_method='min_max',
    alpha=0.05,
    split_method=None,
    n_clusters=None,
    max_clusters=None,
    plot=True,
):
    """
    Calculates series clusters and appends a column of cluster labels to the input df

    df: pandas df
    ts_settings: dictionary of parameters for time series project
    method: type of clustering technique: must choose from either pacf, correlation, performance, or target
    nlags: int (Optional)
        Number of AR(n) lags. Only applies to PACF method
    scale: boolean (Optional)
        Only applies to PACF method
    scale_method: str (Optiona)
        Choose between normalize (subtract the mean and divide by the std) or min_max (subtract the min and divide by the range)
    split_method: str (Optional)
        Choose between rank and quanitles. Only applies to target method
    n_clusters: int
        Number of clusters to create. If None, defaults to maximum silhouette score
    max_clusters: int
        Maximum number of clusters to create. If None, default to the number of series - 1

    Returns:
    --------
    Updated pandas df with a new column 'Cluster' of clusters labels
            -silhouette score per cluster:
            (The best value is 1 and the worst value is -1. Values near 0 indicate overlapping
            clusters. Negative values generally indicate that a sample has been assigned to the
            wrong cluster.)
            -plot of distortion per cluster
    """
    target = ts_settings['target']
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    df = df.copy()

    df.sort_values(by=[series_id, date_col], ascending=True, inplace=True)

    series = df[series_id].unique()
    n_series = len(series)

    if max_clusters is None:
        max_clusters = n_series - 1

    assert (
        1 < max_clusters < n_series
    ), 'max_clusters must be greater than 1 and less than or equal to the number of unique series -1'

    if n_clusters:
        assert (
            1 < n_clusters <= max_clusters
        ), f'n_clusters must be greater than 1 and less than {max_clusters}'

    c = df.pivot(index=date_col, columns=series_id, values=target)

    if method == 'pacf':
        d = pd.DataFrame(
            [_get_pacf_coefs(c, x, nlags, alpha, scale, scale_method) for x in c.columns]
        )  # ignore missing values
        d.index = c.columns
        distances = pdist(d, 'minkowski', p=2)  # 1 for manhattan distance and 2 for euclidean
        dist_matrix = squareform(distances)
        dist_df = pd.DataFrame(dist_matrix)
        dist_df.columns = series
        dist_df.index = dist_df.columns

    elif method == 'correlation':
        dist_df = c.corr(method='pearson')
        dist_df = dist_df.apply(lambda x: x.fillna(x.mean()), axis=1)
        dist_df = dist_df.apply(lambda x: x.fillna(x.mean()), axis=0)

    elif method == 'performance':
        dist_df = _get_performance_cluster_results(df, ts_settings, n_clusters, max_clusters)

    elif method == 'target':
        if split_method is not None:
            if n_clusters:
                cuts = n_clusters
            else:
                cuts = max_clusters

            new_df = _split_series(df, series_id, target, by=split_method, cuts=cuts)
            return new_df  # exit function
        else:
            dist_df = df.groupby(series_id).agg({target: 'mean'})

    else:
        raise ValueError(
            f'{method} is not a supported value. Must be set to either pacf, correlation, performance, or target'
        )

    # Find optimal number of clulsters is n_clusters is not specified
    if n_clusters is None:
        n_clusters = _get_optimal_n_clusters(
            df=dist_df, n_series=n_series, max_clusters=max_clusters, plot=plot
        )

    kmeans = KMeans(n_clusters).fit(dist_df)
    labels = kmeans.predict(dist_df)

    df_clusters = (
        pd.concat([pd.Series(series), pd.Series(labels)], axis=1)
        .sort_values(by=1)
        .reset_index(drop=True)
    )
    df_clusters.columns = [series_id, 'Cluster']

    df_w_cluster_labels = df.merge(df_clusters, how='left', on=series_id)

    return df_w_cluster_labels.reset_index(drop=True)


def plot_clusters(df, ts_settings, split_col='Cluster', max_sample_size=50000):
    """
    df: pandas df
    ts_settings: dictionary of parameters for time series project
    col: cluster_id columns

    Returns:
    --------
    Plotly bar plot

    """
    assert split_col in df.columns, f'{split_col} must be a column in the df'

    date_col = ts_settings['date_col']
    target = ts_settings['target']
    series_id = ts_settings['series_id']

    n_clusters = len(df[split_col].unique())

    if df.shape[0] > max_sample_size:  # limit the data points displayed in the charts to reduce lag
        df = df.sample(n=max_sample_size).reset_index(drop=True)

    df[date_col] = pd.to_datetime(df[date_col])
    df.sort_values(by=[split_col, date_col], inplace=True)
    df_agg = df.groupby([split_col, date_col]).agg({target: 'mean'}).reset_index()
    groups = df_agg.groupby([split_col])
    fig = make_subplots(rows=n_clusters, cols=1)

    a = 1
    for name, group in groups:
        n_series = len(df.loc[df[split_col] == name, series_id].unique())
        fig.append_trace(
            go.Line(
                x=group[date_col], y=group[target], name=f'{split_col}={name} - {n_series} Series'
            ),
            row=a,
            col=1,
        )
        a += 1

    fig.update_layout(height=1000, width=1000, title_text="Cluster Subplots")
    fig.show()

    
def reshape_df(df, ts_settings, agg_level= 'W', scale= False):
    """
    Restructures a dataset for use in dimensionality reduction

    df: Pandas DataFrame
        Input dataframe with time series data
    ts_settings: dict
        Pre-defined time series projet settings
    agg_level: str
        Resampling frequency, allowed values found in pandas docs: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.resample.html
    scale: bool
        True / False. Controls if output df is MinMax scaled
    Returns:
    --------
    Pandas DataFrame
    """
    
    output = pd.DataFrame()
    dates = set()
    for series in df[ts_settings['series_id']].unique().tolist():
        data = df[df[ts_settings['series_id']] == series]
        data = data.set_index(ts_settings['date_col'])
        data = data.resample(agg_level).sum()
        dates.update([x.date() for x in data.index.tolist()]) # [x.date() for x in data['index'].tolist()]
        data[ts_settings['series_id']] = series
        output = output.append(data)
    output = output.groupby(ts_settings['series_id'])[ts_settings['target']].apply(lambda df: df.reset_index(drop= True)).unstack().fillna(0)
    output.columns = sorted(list(dates))
    idx = output.index
    if scale:
        output = pd.DataFrame(MinMaxScaler(feature_range=(0, 5)).fit_transform(output), columns= sorted(list(dates)), index= idx)
    return output
    
    
    
def plot_UMAP(df_T, df_clustered, ts_settings):
    """
    Perform dimensionality reduction and plot a transformed dataframe to assess clustering efficacy

    df_T: Pandas DataFrame
        Transposed dataframe for dimensionality reduction
    df_clustered: Pandas DataFrame
        Dataframe with series_id and cluster labels
    ts_settings: dict
        Pre-defined time series projet settings

    Returns:
    --------
    Plotly 3D scatter plot
    """
    
    pca = umap.UMAP(n_neighbors=10, 
                    n_components=3, 
                    metric='euclidean', 
                    learning_rate=1.0, 
                    init='spectral', 
                    min_dist=0.5, 
                    spread=1.0,  
                    set_op_mix_ratio=1.0,
                    local_connectivity=1.0, 
                    repulsion_strength=1.0, 
                    negative_sample_rate=5, 
                    transform_queue_size=4.0,  
                    angular_rp_forest=False, 
                    target_n_neighbors=-1, 
                    target_metric='categorical', 
                    target_weight=0.5 
                    ) # output_metric='euclidean',  
    principalComponents = pca.fit_transform(df_T)
    embedding = pd.DataFrame(data = principalComponents, columns = ['PC_1', 'PC_2', 'PC_3'])
    labels = pd.DataFrame(df_clustered.groupby(ts_settings['series_id'])['Cluster'].max().reset_index())
    embedding['label'] = df_T.index
    principalDf = pd.merge(embedding, labels, how= 'inner', left_on= embedding['label'], right_on= labels[ts_settings['series_id']])
    
    fig = px.scatter_3d(principalDf, x='PC_1', y='PC_2', z='PC_3', color='Cluster', hover_data=['Cluster'],
                       width=500, height=500)
    fig.show()