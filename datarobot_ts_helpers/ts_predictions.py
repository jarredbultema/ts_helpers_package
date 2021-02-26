import datarobot as dr
import pandas as pd
import numpy as np

from .ts_projects import get_top_models_from_projects


###############
# Predictions
###############
def series_to_clusters(df, ts_settings, split_col='Cluster'):
    '''
    Creates a series map corresponding to series clusters
    
    df: pandas df
    ts_settings: dict
        Parameters for time series project
    split_col: str
        Column name in df to be used to subset data
    
    Returns:
    --------
    dict
    '''
    
    series_id = ts_settings['series_id']

    series = df[[series_id, split_col]].drop_duplicates().reset_index(drop=True)
    series_map = {k: str(v) for (k, v) in zip(series[series_id], series[split_col])}
    return series_map


def clusters_to_series(df, ts_settings, split_col='Cluster'):
    '''
    Creates a cluster map corresponds to series within a cluster
    
    df: pandas df
    ts_settings: dict
        Parameters for time series project
    split_col: str
        Column name in df to be used to subset data
    
    Returns:
    --------
    dict
    '''
    
    series_id = ts_settings['series_id']

    df = df[[series_id, split_col]].drop_duplicates().reset_index(drop=True).sort_values([split_col, series_id], ascending= [True, True])
    groups = df.groupby(split_col)[series_id].apply(lambda x: [i for i in x])
    if min([int(x) for x in df[split_col].unique()]) == 0:
        clusters_to_series = {str(i): g for i, g in enumerate(groups)}
    else:
        clusters_to_series = {str(i): g for i, g in enumerate(groups,1)}
    return clusters_to_series


def get_project_stats(
    projects, n_models, cluster_to_series_map, models= None, metric=None, split_col='Cluster', prefix='TS', data_subset= 'allBacktests'
):
    '''
    projects: list
        list of DataRobot model objects
    n_models: int
        number of models to select from each DR project
    cluster_to_series_map: dict
        Dictionary to remap series and clusters
    models: list (optional)
        List of DataRobot datetime models to be used. If None, top model will be used
    metric: str
        Metric to be used for sorting the leaderboard, if None uses project metric
    split_col: str
        Column name in df to be used to subset data
    prefix: str
        Label to append to project name
     data_subset: str (optional)
        Can be set to either allBacktests or holdout
    
    Returns:
    --------
    Pandas df
    '''
    
    stats = pd.DataFrame()
    for i, p in enumerate(projects):
        if metric is None:
            metric = p.metric

        project_name_char = prefix + '_FD:'

        if split_col is not None:
            split_col_char = '_' + split_col + '-'
            stats.loc[i, 'Project_Name'] = p.project_name
            stats.loc[i, 'Project_ID'] = p.id
            stats.loc[i, split_col] = p.project_name.split(split_col_char)[1]
            stats.loc[i, 'FD'] = p.project_name.split(project_name_char)[1].split('_FDW:')[0]
            stats.loc[i, 'FDW'] = p.project_name.split('_FDW:')[1].split(split_col_char)[0]

        else:
            stats.loc[i, 'Project_Name'] = p.project_name
            stats.loc[i, 'Project_ID'] = p.id
            stats.loc[i, split_col] = None
            stats.loc[i, 'FD'] = p.project_name.split(project_name_char)[1].split('_FDW:')[0]
            stats.loc[i, 'FDW'] = p.project_name.split('_FDW:')[1].replace("_all_series","")


        # stats.loc[i, 'Project_Name'] = p.project_name
        # stats.loc[i, 'Project_ID'] = p.id
        # stats.loc[i, split_col] = p.project_name.split(split_col_char)[1]
        # stats.loc[i, 'FD'] = p.project_name.split(project_name_char)[1].split('_FDW:')[0]
        # stats.loc[i, 'FDW'] = p.project_name.split('_FDW:')[1].split(split_col_char)[0]

        if models is not None:
            m = models[i]
        elif models is None:
            m = get_top_models_from_projects([p], n_models=1, metric=metric, data_subset= data_subset)[0]
        else:
            print('There is an issue with selection of a model')
        stats.loc[i, 'Model_Type'] = m.model_type
        stats.loc[i, 'Model_ID'] = m.id

    if split_col is not None:
        stats['Series'] = stats[split_col].map(cluster_to_series_map)
    else:
        stats['Series'] = None
    return stats



def deploy_models(models,
                  labels=None,
                  descriptions=None,
                  pred_server=None):
    '''
    Deploy a list of DataRobot models

    models: list
        list of DataRobot model objects to deploy
    labels: list (optional)
        list of str for title of deployments
    descriptions: list (optional)
        list of str for description for deployments
    pred_server: datarobot.models.prediction_server.PredictionServer (optional)
        DataRobot prediction server object, or None and will automatically retrieve the first option

    Returns:
    --------
    deployments: list
    '''

    assert isinstance(models, list), 'models must be a list of DataRobot models'
    if labels is not None:
        assert isinstance(labels, list), 'labels must be a list of desired Deployment titles or None'
        assert len(models) == len(labels), 'labels must contain the same number of entries as models to be deployed'
    if descriptions is not None:
        assert isinstance(descriptions, list), 'descriptions must be a list of desired Deployment descriptions or None'
        assert len(models) == len(
            descriptions), 'descriptions must contain the same number of entries as models to be deployed'
    if pred_server is not None:
        assert isinstance(pred_server,
                          datarobot.models.prediction_server.PredictionServer), 'pred_server must be a datarobot.models.prediction_server.PredictionServer object or None'

    deployments = []

    # get values if not supplied
    if labels is None:
        labels = [dr.Project.get(x.project_id).project_name for x in models]
    if descriptions is None:
        descriptions = [f'Deployment of best model from {dr.Project.get(x.project_id).project_name}' for x in models]
    if pred_server is None:
        pred_server = dr.PredictionServer.list()[0]

    # create deployment for each model
    for m, l, d in list(zip(models, labels, descriptions)):
        try:
            deployments.append(dr.Deployment.create_from_learning_model(m.id, label=l, description=d,
                                                                        default_prediction_server_id=pred_server.id))
            print(f'Deployment of {m.model_type} into {dr.Project.get(m.project_id).project_name} successful!')
        except:
            print(f'*** Something went wrong when deploying {m.model_type} ***')

    # return list of deployment objects
    return deployments


# version from AFRICOM
def get_or_request_predictions(
    models,
    scoring_df,
    training_df,
    ts_settings,
    deployments= None,
    project_stats=None,
    start_date=None,
    end_date=None,
    forecast_point=None,
    retrain=False,
):
    '''
    models: list
        list of DataRobot datetime project objects
    deployments: list
        list of DataRobot deployment ids
    scoring_df: pandas df
        Predictions dataframe that contains required information (KIA, future datetime stamp, etc) correspond to a desired range of predictions
    training_df: pandas df (optional)
        Predictions dataframe that contains training data used to build the model, required to augment FDW data
    ts_settings: dict
        Parameters for the time series projects in DR
    project_sats: pandas df
        output of get_project_stats(), contains detailed information on DR projects
    start_date: datetime
        Desired start date for DR project retraining from a frozen model
    end_date: datetime
        Desired end date for DR project retraining from a frozen model
    forecast_point: datetime
        Desired forecast point for start of predictions, must be configured associated with scoring_df
    retrain: bool
        Controls if a frozen DR datetime model will be retrained on a new training period
        
    Returns:
    --------
    pandas df
    '''
    
    series_id = ts_settings['series_id']
    date = ts_settings['date_col']
    training_df[date] = pd.to_datetime(training_df[date])#.apply(lambda x: x.date())
    scoring_df[date] = pd.to_datetime(scoring_df[date])#.apply(lambda x: x.date())

    models_to_predict_on = []
    retrain_jobs = []
    predict_jobs = []
    project_dataset_map = {}

    if deployments is not None:
        models = []
        for d in deployments:
            print(f'Accessing model from {d.label} deployment')
            models.append(dr.DatetimeModel.get(project= d.model['project_id'], model_id= d.model['id']))

    for m in models:
        print(f"\n{m}")
        p = dr.Project.get(m.project_id)
        series = project_stats.loc[project_stats['Model_ID'] == m.id, 'Series'].values
        if len(series) == 1:
                series = series[0]
        elif series.sort() != series.sort():
            print(f'There are no suitable series in {p} to for predctions')
            continue
        elif len(series) == 0:
            print(f'There are no suitable series in {p} to for predctions')
            continue
        
        print('For series: ', series)

        if isinstance(series,list):
            data = scoring_df[scoring_df[series_id].isin(series)]

        else:
            data = scoring_df

        if data.shape[0] == 0:
            print(f'*** There are no rows to score in {p.project_name} using {start_date} ... ***\n')
            continue

        start = m.training_info['prediction_training_start_date'] # 
#         start = dr.DatetimePartitioning.get(p.id).holdout_start_date # m.training_info['prediction_training_start_date'] # 
        end = m.training_info['prediction_training_end_date'] # 
#         end= dr.DatetimePartitioning.get(p.id).holdout_end_date # m.training_info['prediction_training_end_date'] # 
        if start_date is None:
            start_date = pd.to_datetime(start)
        if end_date is None:
            end_date = pd.to_datetime(end)
        if forecast_point is None:
            pred_date = data[date].min()
        else:
            pred_date = pd.to_datetime(forecast_point)
        cutoff = (pd.to_datetime(pred_date) + pd.DateOffset(int(project_stats[project_stats['Model_ID'] == m.id]['FDW'].values[0])))

        print('Training Data Start Date: ', start_date)
        print('Training Data End Date: ', end_date)
        print('FDW start for predictions: ', cutoff)
        print('FDW end for predictions: ', (pred_date - pd.DateOffset(int(ts_settings['fdw_end']))))
        print('Forecast point: ', pred_date)

        if training_df is not None and data[date].min() > cutoff:
            if series is not None:
                historical_data = training_df[(training_df[date] >= cutoff) & (training_df[series_id].isin(series))]
            else:
                historical_data = training_df[(training_df[date] >= cutoff)]

            df = pd.concat((historical_data, data), axis=0).drop_duplicates([series_id, date])
            print('*** Values required to generate FDW for predictions are missing from the scoring_df and will be added from the training_df ***')
            print(f'Adding historical data from {historical_data[date].min()} to {data[date].min()}')
        else:
            df = data # .drop_duplicates([series_id, date])

        print(f'Uploading scoring dataset with {df.shape[0]} rows for Project {p.project_name}')
        
        if any(df[ts_settings['known_in_advance']].isnull().any(axis=1)):
            df[ts_settings['known_in_advance']] = df[ts_settings['known_in_advance']].fillna(method= 'ffill')
            if any(df[ts_settings['known_in_advance']].isnull().any(axis=1)):
                df[ts_settings['known_in_advance']] = df[ts_settings['known_in_advance']].fillna(method= 'bfill')
                print('*** Missing values were detected in the KIA variables, and were filled using backward-fill ***')
            else:
                print('*** Missing values were detected in the KIA variables, and were filled using forward-fill ***')

        # only upload if necessary
        if m.project_id not in project_dataset_map:
            p.unlock_holdout()
            pred_dataset = p.upload_dataset(df, forecast_point=forecast_point)
            project_dataset_map[m.project_id] = pred_dataset.id

        if retrain:
            try:
                new_model_job = m.request_frozen_datetime_model(
                    training_start_date=start_date, training_end_date=end_date
                )
                retrain_jobs.append(new_model_job)
                print(
                    f'Retraining M{m.model_number} from {start_date.date()} to {end_date.date()} in Project {p.project_name}'
                )
            except dr.errors.JobAlreadyRequested:
                print(
                    f'M{m.model_number} in Project {p.project_name} has already been retrained through the holdout'
                )
                models_to_predict_on.append(m)
        else:
            models_to_predict_on.append(m)

        for job in retrain_jobs:
            models_to_predict_on.append(job.get_result_when_complete(max_wait=10000))

    for model in models_to_predict_on:
        p = dr.Project.get(model.project_id)
        print(f'Getting predictions for M{model.model_number} in Project {p.project_name}')
        predict_jobs.append(model.request_predictions(project_dataset_map[model.project_id]))

    preds = [pred_job.get_result_when_complete() for pred_job in predict_jobs]

    predictions = pd.DataFrame()
    for i in range(len(preds)):
        predictions = predictions.append(preds[i])

    print('\nFinished computing and downloading predictions')
    return predictions



def merge_preds_and_actuals(preds, actuals, ts_settings):
    '''
    Combined actuals from training data along with model predictions into a single df
    
    preds: pandas df
        output from get_or_request_predictions(), contains model predictions from a defined period
    actuals: pandas df
        pandas df containing training data
    ts_settings: dict
        Parameters for a time series DR project
        
    Returns:
    --------
    pandas df
    '''
    
    series_id = ts_settings['series_id']
    date_col = ts_settings['date_col']

    actuals[date_col] = pd.to_datetime(actuals[date_col]).dt.tz_localize(None)
    preds['timestamp'] = pd.to_datetime(preds['timestamp']).dt.tz_localize(None)
    preds_and_actuals = preds.merge(
        actuals, how='left', left_on=['series_id', 'timestamp'], right_on=[series_id, date_col]
    )
    return preds_and_actuals

