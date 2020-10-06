import datarobot as dr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean 
import re


def get_top_models_from_project(
    project, n_models=1, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    project: project object
        DataRobot project
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    List of model objects from a DataRobot project

    """
    assert data_subset in [
        'backtest_1',
        'allBacktests',
        'holdout',
    ], 'data_subset must be either backtest_1, allBacktests, or holdout'
    if n_models is not None:
        assert isinstance(n_models, int), 'n_models must be an int'
    if n_models is not None:
        assert n_models >= 1, 'n_models must be greater than or equal to 1'
    assert isinstance(include_blenders, bool), 'include_blenders must be a boolean'

    mapper = {
        'backtest_1': 'backtestingScores',
        'allBacktests': 'backtesting',
        'holdout': 'holdout',
    }

    if metric is None:
        metric = project.metric

    if data_subset == 'holdout':
        project.unlock_holdout()

    models = [
        m
        for m in project.get_datetime_models()
        if m.backtests[0]['status'] != 'BACKTEST_BOUNDARIES_EXCEEDED'
    ]  # if m.holdout_status != 'HOLDOUT_BOUNDARIES_EXCEEDED']

    if data_subset == 'backtest_1':
        # models = sorted(models, key=lambda m: np.mean([i for i in m.metrics[metric][mapper[data_subset]][0] if i]), reverse=False)
        models = sorted(
            models, key=lambda m: m.metrics[metric][mapper[data_subset]][0], reverse=False
        )
    elif data_subset == 'allBacktests':
        models = sorted(
            models,
            key=lambda m: m.metrics[metric][mapper[data_subset]]
            if m.metrics[metric][mapper[data_subset]] is not None
            else np.nan,
            reverse=False,
        )
    else:
        try: 
            models = sorted(models, key=lambda m: m.metrics[metric][mapper[data_subset]], reverse=False)
            
        except:
            return f'This project does not have an appropriate {data_subset} configured'

    if not include_blenders:
        models = [m for m in models if m.model_category != 'blend']

    if n_models is None:
        n_models = len(models)

    models = models[0:n_models]

    assert len(models) > 0, 'You have not run any models for this project'

    return models


def get_backtest_information(
    p, models, entry, entry_count, ts_settings
):
    """
    Get training and backtest durations from a model from one DataRobot project

    p: datarobot.models.project.Project
        DataRobot project object
    entry: list
        DataRobot model backtest information
    entry_count: int/str
        Counter for backtest number, or designation as holdout
    ts_settings: dict
        Parameters for time series project

    Returns:
    --------
    list
    """
    
    backtest_name = f'backtest_{entry_count}'
    if not isinstance(entry_count, int):
        backtest_name = 'holdout'
    
    training_duration = re.search('\d*',entry['training_duration']).group(0)    # .lstrip('P').rstrip('D')
    training_start = pd.to_datetime(entry['training_end_date'].date())
    training_end = pd.to_datetime(entry['training_start_date'].date())
    validation_start = pd.to_datetime(entry['training_start_date'].date()) + pd.Timedelta(days= ts_settings['fd_start'])
    validation_end = validation_start + pd.Timedelta(days=ts_settings['validation_duration'])
    return [p, models[0], backtest_name, training_start, training_end, training_duration,  validation_start, validation_end]
        


def get_training_and_backtest_windows(
    projects, ts_settings, data_subset='allBacktests', metric= None
):
    """
    Get training and backtest durations from models across multiple DataRobot projects

    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    data_subset: str (optional)
        Can be set to either allBacktests, backtest_n (n= Backtest number), holdout
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    pandas df
    """
    
    assert isinstance(projects, list), 'Projects must be a list object'

    durations = pd.DataFrame()
    df_columns = ['DR project', 'DR model', 'backtest id', 'training start date','training end date'\
                  , 'duration', 'validation start date', 'validation end date']

    backtest_error_count = 0
    holdout_error_count = 0
    
    print('Getting backtest information for all projects...')
    
    for p in projects:
        
        if metric is None:
            metric = p.metric
            
        project_data = []

        if data_subset == 'allBacktests':
            models = get_top_models_from_project(
            p,
            data_subset=data_subset,
            include_blenders=False,
            metric=metric,
            )
            
            backtest = models[0].backtests
            for idx, entry in enumerate(backtest,1):
                project_data.append(get_backtest_information(p, models, entry, idx, ts_settings))

        elif re.search('backtest_*', data_subset):
            models = get_top_models_from_project(
            p,
            data_subset='allBacktests',
            include_blenders=False,
            metric=metric,
            )
            
            if int(data_subset[-1]) > len(models[0].backtests):
                   return f'There are not {data_subset[-1]} backtests in this project. Please select a lower value.'
            
            backtest = models[0].backtests[int(data_subset[-1])-1]
            
            project_data.append(get_backtest_information(p, models, backtest, int(data_subset[-1]), ts_settings))

        elif data_subset == 'all':
            if metric is None:
                metric = p.metric

            try:
                all_backtests = get_training_and_backtest_windows([p], ts_settings, data_subset= 'allBacktests', metric= metric)
                durations = pd.concat((durations,pd.DataFrame(all_backtests, columns= df_columns)), axis=0)      
            except:
                backtest_error_count += 1
            
            try:
                holdout_data = get_training_and_backtest_windows([p], ts_settings, data_subset= 'holdout', metric= metric)
                durations = pd.concat((durations,pd.DataFrame(holdout_data, columns= df_columns)), axis=0)     
            except:
                holdout_error_count += 1
    
        elif data_subset == 'holdout':
            models = get_top_models_from_project(
            p,
            data_subset=data_subset,
            include_blenders=False,
            metric=metric,
            )
            
            assert isinstance(models, list), 'holdout not configured for these projects'
            
            backtest = models[0].backtests
            project_data.append(get_backtest_information(p, models, backtest, data_subset, ts_settings))
    
        else:
            return "Only data_subset values of 'allBacktests', 'backtest_n' where n = backtest number, or 'holdout' are allowed"  
            
        durations = pd.concat((durations,pd.DataFrame(project_data, columns= df_columns)), axis=0)
        
    if backtest_error_count > 0:
        print(f'***** There were errors with backtests configuration in {backtest_error_count} projects. That data omitted *****\n')
    if holdout_error_count > 0:
        print(f'***** There were errors with holdout configuration in {holdout_error_count} projects. That data omitted *****\n')
    
    return durations.fillna(0)


def check_series_backtests(cluster_information, series_name, ts_settings, backtest_information):
    """
    Determines series-level coverage across multiple backtests

    cluster_information: pandas df
        Information about each series including a cluster id, output from add_cluster_labels()
    series_name: str
        Name of an individual series
    ts_settings: dict
        Parameters for time series project
    backtest_information: pandas df
        contains information on how many records are present for each series in each backtest 
        , output from get_training_and_backtest_windows()

    Returns:
    --------
    Pandas DataFrame
    """
    series_dates = cluster_information[cluster_information[ts_settings['series_id']] == series_name][ts_settings['date_col']]
    cluster_id = cluster_information[cluster_information[ts_settings['series_id']] == series_name]['Cluster'].unique().tolist()[0]

    if all(backtest_information['DR project'].astype(str).str.contains('_all_series')):
        single_cluster = True
    else:
        single_cluster = False 
        
    if 0 in cluster_information['Cluster'].unique().tolist():
        cluster_id += 1

    present = []
    absent = []
    
    if single_cluster:
        for test in backtest_information['backtest id'].unique().tolist():
            start = backtest_information[backtest_information['backtest id'] == test]['validation start date'].tolist()[0]
            end = backtest_information[backtest_information['backtest id'] == test]['validation end date'].tolist()[0] - pd.DateOffset(1)
            if any(series_dates.between(start, end)):
                present.append((test,np.sum(series_dates.between(start, end))))
            if not any(series_dates.between(start, end)):
                absent.append(test)    
    else:
        cluster_data = backtest_information[backtest_information['DR project'].astype(str).str.contains(f'_Cluster-{cluster_id}')]
        for test in backtest_information['backtest id'].unique().tolist():
            try:
                start = cluster_data[cluster_data['backtest id'] == test]['validation start date'].tolist()[0]
                end = cluster_data[cluster_data['backtest id'] == test]['validation end date'].tolist()[0] - pd.DateOffset(1)
            except:
                absent.append(test)
                continue

            if any(series_dates.between(start, end)):
                present.append((test,np.sum(series_dates.between(start, end))))
            if not any(series_dates.between(start, end)):
                absent.append(test)
                
    return present, absent


def check_all_series_backtests(cluster_information, ts_settings, backtest_information):
    """
    Plots series-level coverage across multiple backtests

    cluster_information: pandas df
        Information about each series including a cluster id, output from add_cluster_labels()
    ts_settings: dict
        Parameters for time series project
    backtest_information: pandas df
        contains information on how many records are present for each series in each backtest 
        , output from get_training_and_backtest_windows()

    Returns:
    --------
    Pandas DataFrame
    """
    
    df = pd.DataFrame([], columns= backtest_information['backtest id'].unique().tolist(), index= cluster_information[ts_settings['series_id']].unique().tolist())
    for series in df.index.tolist():
        present, absent = check_series_backtests(cluster_information, series, ts_settings, backtest_information)
        df.loc[series] = dict(present)
    
    return df.fillna(0).astype(int)


def get_series_in_backtests(df, data_subset, present= True, threshold= None):
    """
    Selects the subset of series that are present or absent in any defined backtest
    df: Pandas df
        Output of check_all_series_backtests(), contains information on presence of series in each backtest period
    data_subset: str
        Which data_subsets should be included in analysis, accpets individual backtests ('backtest_1', 'allBacktests', 'holdout')
    present: bool
        Select series that are present (True) or absent (False) from backtesting window(s)
    threshold: np.float (0.0 - 1.0)
        cutoff threshold to determine presence
        
    Returns:
    --------
    series: list
        Series names that match the selection conditions
    """
    
    avail_backtests = df.columns.tolist()[1:]
    if data_subset.lower() == 'allbacktests':
        select_backtest = avail_backtests
    else:   
        assert data_subset in [avail_backtests], 'data_subset must be present in input df'
        select_backtest = data_subset.lower()
    
    
    cutoff = 0
    if threshold is not None:
        cutoff = int(df[select_backtest].max().values.max() * threshold)
    if present:
        print(f'Getting series with present in {cutoff} or more rows in {", ".join(select_backtest)} ...')

        series = df[(df[select_backtest].T >= cutoff).any()].iloc[:,0].tolist()
    else:
        print(f'Getting series with present in {cutoff} or fewer rows rows in {", ".join(select_backtest)} ...')
        if cutoff == 0:
            series = df[(df[select_backtest].T == cutoff).any()].iloc[:,0].tolist()
        else:
            series = df[(df[select_backtest].T < cutoff).any()].iloc[:,0].tolist()
    
    return series


def plot_series_backtest_coverage(series_backtests, ts_settings, n=50):
    """
    Plots series-level coverage across multiple backtests

    series_backtests: pandas df
        Output from check_all_series_backtests()
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Choose from either holdout or allBacktests
    n: int
        Number of series to display

    Returns:
    --------
    Plotly barplot
    """
    n_series = len(series_backtests.index.tolist())
    n = min(n_series, n)
    series_backtests.reset_index(inplace= True)
    series_backtests = series_backtests.sort_values('index') # [0:n,:]

    fig = go.Figure(data= [
        go.Bar(name='backtest 1', x=series_backtests['index'], y=series_backtests['backtest_1']),
        go.Bar(name='backtest 2', x=series_backtests['index'], y=series_backtests['backtest_2']),
        go.Bar(name='backtest 3', x=series_backtests['index'], y=series_backtests['backtest_3'])
    ])
    
    fig.update_layout(barmode='group', title_text=f'Series Presence in Backtests', height= 400)
    fig.update_yaxes(title='Records present in backtest')
    fig.update_xaxes(tickangle=45)
    fig.show()
    
    
if __name__ == '__main__':
    
    # get DR project(s)
    projects = [x for x in dr.Project.list() if 'tag' in str(x) and x.stage == 'modeling']
    
    # Set default values
    target= 'sale_amount_sum'
    date_col = 'date'
    series_id = 'item_name'
    kia = None # No columns known in advance for this dataset!
    num_backtests = 3
    validation_duration = 30 # want to predict 1-monath sales
    holdout_duration = 30 
    disable_holdout = False
    metric = 'RMSE' # what makes most sense in this case?
    use_time_series = True
    fd_start = 1
    fd_end = 31 # forecasting sales for the next month
    fdw_start = -28 # we should iterate on this
    fdw_end = 0
    max_date = df_w_clusters['date'].max()

    # create Time Series settings
    ts_settings = {'max_date':max_date, 'known_in_advance':kia, 'num_backtests':num_backtests, 
                   'validation_duration':validation_duration, 'holdout_duration':holdout_duration,
                   'disable_holdout':disable_holdout,'use_time_series':use_time_series,
                   'series_id':series_id, 'metric':metric, 'target':target, 'date_col':date_col,
                   'fd_start':fd_start, 'fd_end':fd_end, 'fdw_start':fdw_start, 'fdw_end':fdw_end}
    
    # calculate how many records are present for all series in backtests
    
    projects_backtesting_info = ts.get_training_and_backtest_windows(projects, ts_settings, data_subset='allBacktests', metric= None)
    projects_backtests = ts.check_all_series_backtests(training_dataset, ts_settings, projects_backtesting_info)
    ts.plot_series_backtest_coverage(projects_backtests, ts_settings, n=50)