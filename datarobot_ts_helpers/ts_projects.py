import datarobot as dr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statistics import mean 
import re

from .ts_metrics import *
from .ts_data_quality import *
from .ts_data_quality import _cut_series_by_rank
from .ts_modeling import *


######################
# Project Evaluation
######################


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


def get_top_models_from_projects(
    projects, n_models=1, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    Pull top models from leaderboard across multiple DataRobot projects

    projects: list
        DataRobot project object(s)
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    List of model objects from DataRobot project(s)
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    models_all = []
    for p in projects:
        models = get_top_models_from_project(p, n_models, data_subset, include_blenders, metric)
        models_all.extend(models)
    return models_all


def get_ranked_model(project, model_rank, metric= None, data_subset= 'allBacktests'):
    """
    project: project object
        DataRobot project
    model_rank: int
        None if top model, model leaderboard rank if any model other than top desired
    metric: str (optional)
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    data_subset: str (optional)
        Can be set to either backtest_1, allBacktests or holdout

    Returns:
    --------
    model object from a DataRobot project
    """
    assert data_subset in [
        'backtest_1',
        'allBacktests',
        'holdout',
    ], 'data_subset must be either backtest_1, allBacktests, or holdout'

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

    models = models[model_rank -1: model_rank]
    
    if any([m.model_category is 'blend' for m in models]):
        print('Blenders cannot be retrained on reduced feature lists')

    if not models:
        print('You have not run any models for this project')

    return models


def compute_backtests(
    projects, n_models=5, data_subset='backtest_1', include_blenders=True, metric=None
):
    """
    Compute all backtests for top models across multiple DataRobot projects

    projects: list
        DataRobot project object(s)
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    """
    assert isinstance(projects, list), 'Projects must be a list object'

    for p in projects:
        models = get_top_models_from_project(
            p,
            n_models=n_models,
            data_subset=data_subset,
            include_blenders=include_blenders,
            metric=metric,
        )

        for m in models:
            try:
                m.score_backtests()  # request backtests for top models
                print(f'Computing backtests for model {m.id} in Project {p.project_name}')
            except dr.errors.ClientError:
                pass
        print(
            f'All available backtests have been submitted for scoring for project {p.project_name}'
        )


def get_or_request_backtest_scores(
    projects, n_models=5, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    Get or request backtest and holdout scores from top models across multiple DataRobot projects

    projects: list
        DataRobot project object(s)
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    pandas df
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    scores = pd.DataFrame()
    for p in projects:

        models = get_top_models_from_project(
            p,
            n_models=n_models,
            data_subset=data_subset,
            include_blenders=include_blenders,
            metric=metric,
        )

        if metric is None:
            metric = p.metric

        backtest_scores = pd.DataFrame(
            [
                {
                    'Project_Name': p.project_name,
                    'Project_ID': p.id,
                    'Model_ID': m.id,
                    'Model_Type': m.model_type,
                    'Featurelist': m.featurelist_name,
                    f'Backtest_1_{metric}': m.metrics[metric]['backtestingScores'][0],
                    'Backtest_1_MASE': m.metrics['MASE']['backtestingScores'][0],
                    'Backtest_1_Theils_U': m.metrics["Theil's U"]['backtestingScores'][0],
                    'Backtest_1_SMAPE': m.metrics['SMAPE']['backtestingScores'][0],
                    'Backtest_1_R_Squared': m.metrics['R Squared']['backtestingScores'][0],
                    f'All_Backtests_{metric}': m.metrics[metric]['backtestingScores'],
                    'All_Backtests_MASE': m.metrics['MASE']['backtestingScores'],
                    'All_Backtests_Theils_U': m.metrics["Theil's U"]['backtestingScores'],
                    'All_Backtests_SMAPE': m.metrics['SMAPE']['backtestingScores'],
                    'All_Backtests_R_Squared': m.metrics['R Squared']['backtestingScores'],
                    f'Holdout_{metric}': m.metrics[metric]['holdout'],
                    'Holdout_MASE': m.metrics['MASE']['holdout'],
                    'Holdout_Theils_U': m.metrics["Theil's U"]['holdout'],
                    'Holdout_SMAPE': m.metrics['SMAPE']['holdout'],
                    'Holdout_R_Squared': m.metrics['R Squared']['holdout'],
                }
                for m in models
            ]
        ).sort_values(by=[f'Backtest_1_{metric}'])

        scores = scores.append(backtest_scores).reset_index(
            drop=True
        )  # append top model from each project

    print(f'Scores for all {len(projects)} projects have been computed')

    return scores


def get_or_request_training_predictions_from_model(model, data_subset='allBacktests'):
    """
    Get row-level backtest or holdout predictions from a model

    model: DataRobot Datetime model object
        DataRobot project object(s)
    data_subset: str (optional)
        Can be set to either allBacktests or holdout

    Returns:
    --------
    pandas Series
    """

    project = dr.Project.get(model.project_id)

    if data_subset == 'holdout':
        project.unlock_holdout()

    try:
        predict_job = model.request_training_predictions(data_subset)
        training_predictions = predict_job.get_result_when_complete(max_wait=10000)
       
    except dr.errors.ClientError:
        prediction_id = [
            p.prediction_id
            for p in dr.TrainingPredictions.list(project.id)
            if p.model_id == model.id and p.data_subset == data_subset
        ][0]
        training_predictions = dr.TrainingPredictions.get(project.id, prediction_id)
    
    return training_predictions.get_all_as_dataframe() # serializer='csv'


def get_or_request_training_predictions_from_projects(
    projects, models = None, n_models=1, data_subset='allBacktests', include_blenders=True, metric=None
):
    """
    Get row-level backtest or holdout predictions from top models across multiple DataRobot projects

    projects: list
        DataRobot project object(s)
    models: list of DataRobot datetime model or None (optional)
        Model to be used for predictions, if None, top model will be used
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    pandas Series
    """
    assert isinstance(projects, list), 'Projects must be a list object'

    preds = pd.DataFrame()
    
    for idx,p in enumerate(projects):

        if models is None:
            models_p = get_top_models_from_project(p, n_models, data_subset, include_blenders, metric)
        else:
            models_p = [models[idx]]

        for m in models_p:
            tmp = get_or_request_training_predictions_from_model(m, data_subset)
            tmp['Project_Name'] = p.project_name
            tmp['Project_ID'] = p.id
            tmp['Model_ID'] = m.id
            tmp['Model_Type'] = m.model_type
        preds = preds.append(tmp).reset_index(drop=True)
    
    return preds


def get_preds_and_actuals(
    df,
    projects,
    ts_settings,
    models= None,
    n_models=1,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
):
    """
    Get row-level predictions and merge onto actuals

    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    models: list or None (optional)
        List of DataRobot datetime models to be used for predictions. If None, top model will be used
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    pandas df
    """
    assert isinstance(projects, list), 'Projects must be a list object'
    if models is not None:
        assert isinstance(models, list), 'If models is not None, it must be a list of model objects'

    series_id = ts_settings['series_id']

    preds = get_or_request_training_predictions_from_projects(
        projects,
        models= models,
        n_models=1,
        data_subset=data_subset,
        include_blenders=include_blenders,
        metric=metric,
    )
    preds['timestamp'] = pd.to_datetime(preds['timestamp'].apply(lambda x: x[:-8]))
    df = df.merge(
        preds,
        how='left',
        left_on=[ts_settings['date_col'], ts_settings['series_id']],
        right_on=['timestamp', 'series_id'],
        validate='one_to_many',
    )
    df = df.loc[~np.isnan(df['prediction']), :].reset_index(drop=True)
    
    return df


def get_preds_and_actuals_fixed_forecast_point(
    df,
    projects,
    ts_settings,
    forecast_point,
    models= None,
    n_models=1,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
):
    """
    Get row-level predictions and merge onto actuals

    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    forecast_point: str or datetime
        Specific forecast point used for predictions
    models: list or None (optional)
        List of DataRobot datetime models to be used for predictions. If None, top model will be used
    n_models: int
        Number of top models to return
    data_subset: str (optional)
        Can be set to either allBacktests or holdout
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    pandas df
    """
    pa = get_preds_and_actuals(
    df= df,
    projects= projects,
    ts_settings= ts_settings,
    models= models,
    n_models= n_models,
    data_subset= data_subset,
    include_blenders= include_blenders,
    metric= metric)
    forecast_point = pd.to_datetime(forecast_point)
    pa['forecast_point'] = pd.to_datetime(pa['forecast_point']).apply(lambda x: x.date())
    preds_and_actuals = pa[pa['forecast_point'] == pd.to_datetime(forecast_point.date())]
    if preds_and_actuals.shape[0] == 0:
        print('The specified forecast point is not present in the training date, or is incorrectly formatted. Try a str or datetime object')
        return None
    return preds_and_actuals



def get_or_request_model_scores(
    project, model, include_blenders=False, metric=None
):
    """
    Get or request backtest and holdout scores from specified, retrained DataRobot model

    projects: list
        DataRobot project object(s)
    model: dr.Model
        DataRobot DatetimeModel, this is the reference model from which other feature lists were created. 
    include_blenders: boolean (optional)
        Controls whether to include ensemble models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    pandas df
    """

    scores = pd.DataFrame()
    
    if metric is None:
        metric = project.metric

    # select only the models retrained with reduced feature lists
    retrained_models = [x for x in project.get_datetime_models() if str(model).lstrip('[DatetimeModel()').rstrip(')]') in str(x)]

    # calculate backtest scores for new models
    for m in retrained_models:
        try:
            m.score_backtests()
#             print(f'Computing backtests for model {m.id} in Project {project.project_name}')
        except dr.errors.ClientError:
            pass
#             print(f'All available backtests have been submitted for scoring for project {project.project_name}')
    
    backtest_scores = pd.DataFrame(
        [
            {
                'Project_Name': project.project_name,
                'Project_ID': project.id,
                'Model_ID': m.id,
                'Model_Type': m.model_type,
                'Featurelist': m.featurelist_name,
                f'Backtest_1_{metric}': m.metrics[metric]['backtestingScores'][0],
                'Backtest_1_MASE': m.metrics['MASE']['backtestingScores'][0],
                'Backtest_1_Theils_U': m.metrics["Theil's U"]['backtestingScores'][0],
                'Backtest_1_SMAPE': m.metrics['SMAPE']['backtestingScores'][0],
                'Backtest_1_R_Squared': m.metrics['R Squared']['backtestingScores'][0],
                f'All_Backtests_{metric}': m.metrics[metric]['backtestingScores'],
                'All_Backtests_MASE': m.metrics['MASE']['backtestingScores'],
                'All_Backtests_Theils_U': m.metrics["Theil's U"]['backtestingScores'],
                'All_Backtests_SMAPE': m.metrics['SMAPE']['backtestingScores'],
                'All_Backtests_R_Squared': m.metrics['R Squared']['backtestingScores'],
                f'Holdout_{metric}': m.metrics[metric]['holdout'],
                'Holdout_MASE': m.metrics['MASE']['holdout'],
                'Holdout_Theils_U': m.metrics["Theil's U"]['holdout'],
                'Holdout_SMAPE': m.metrics['SMAPE']['holdout'],
                'Holdout_R_Squared': m.metrics['R Squared']['holdout'],
            }
            for m in retrained_models]
        ) # .sort_values(by=[f'Backtest_1_{metric}'])

    scores = scores.append(backtest_scores).reset_index(
        drop=True
    )  
    
    return scores


def get_cluster_acc(
    df,
    projects,
    ts_settings,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
    acc_calc=rmse,
):
    """
    Get cluster-level and overall accuracy across multiple DataRobot projects

    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Valid values are either holdout or allBacktests
    include_backtests: boolean (optional)
        Controls whether blender models are considered
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    acc_calc: function
        Function to calculate row-level prediction accuracy. Choose from mae, rmse, mape, smape, gamma, poission, and tweedie

    Returns:
    --------
    pandas df
    """
    assert isinstance(projects, list), 'Projects must be a list object'
    assert data_subset in [
        'allBacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    print('Getting cluster accuracy...')

    df = get_preds_and_actuals(
        df,
        projects,
        ts_settings,
        n_models=1,
        data_subset=data_subset,
        include_blenders=include_blenders,
        metric=metric,
    )
    df = get_project_info(df)
    groups = (
        df.groupby(['Cluster'])
        .apply(lambda x: acc_calc(x[ts_settings['target']], x['prediction']))
        .reset_index()
    )
    groups.columns = ['Cluster', f'Cluster_{acc_calc.__name__.upper()}']
    groups[f'Total_{acc_calc.__name__.upper()}'] = acc_calc(
        act=df[ts_settings['target']], pred=df['prediction']
    )

    return groups


def plot_cluster_acc(cluster_acc, ts_settings, data_subset='allBacktests', acc_calc=rmse):
    """
    Plots cluster-level and overall accuracy across multiple DataRobot projects

    cluster_acc: pandas df
        Output from get_cluster_acc()
    ts_settings: dict
        Pparameters for time series project
    data_subset: str
        Choose either holdout or allBacktests
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    Plotly barplot
    """
    cluster_acc['Label'] = '=' + cluster_acc['Cluster']

    fig = px.bar(cluster_acc, x='Label', y=f'Cluster_{acc_calc.__name__.upper()}').for_each_trace(
        lambda t: t.update(name=t.name.replace('=', ''))
    )

    fig.add_trace(
        go.Scatter(
            x=cluster_acc['Label'],
            y=cluster_acc[f'Total_{acc_calc.__name__.upper()}'],
            mode='lines',
            marker=dict(color='black'),
            name=f'Overall {acc_calc.__name__.upper()}',
        )
    )

    fig.update_yaxes(title=acc_calc.__name__.upper())
    fig.update_xaxes(tickangle=45)
    fig.update_layout(title_text=f'Cluster Accuracy - {data_subset}')
    fig.show()


def get_series_acc(
    df,
    projects,
    ts_settings,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
    acc_calc=rmse,
):
    """
    Get series-level and overall accuracy across multiple DataRobot projects

    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Valid values are either holdout or allBacktests
    include_backtests: boolean (optional)
        Controls whether blender models are considered
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    acc_calc: function
        Function to calculate row-level prediction accuracy. Choose from mae, rmse, mape, smape, gamma, poission, and tweedie

    Returns:
    --------
    pandas df

    """
    assert isinstance(projects, list), 'Projects must be a list object'
    assert data_subset in [
        'allBacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    series_id = ts_settings['series_id']
    target = ts_settings['target']

    print('Getting series accuracy...')
    
    df = get_preds_and_actuals(
        df,
        projects,
        ts_settings,
        n_models=1,
        data_subset=data_subset,
        include_blenders=include_blenders,
        metric=metric,
    )

    df = get_project_info(df)

    groups = (
        df.groupby([series_id]).apply(lambda x: acc_calc(x[target], x['prediction'])).reset_index()
    )
    groups.columns = [series_id, f'Series_{acc_calc.__name__.upper()}']
    right = df[[series_id, 'Cluster']].drop_duplicates().reset_index(drop=True)
    groups = groups.merge(right, how='left', on=series_id)
    groups[f'Total_{acc_calc.__name__.upper()}'] = acc_calc(act=df[target], pred=df['prediction'])

    return groups


def plot_preds_and_actuals(df, projects, ts_settings, fd_range=None, fd_agg= 'mean', fd= None, average= False, series_name= None, top=None, data_subset= 'allBacktests', include_blenders=False, metric= None, acc_calc=rmse):
    """
    Get series-level and overall accuracy across multiple DataRobot projects

    df: pandas df
    projects: list
        DataRobot project object(s)
    ts_settings: dict
        Parameters for time series project
    fd_range: tuple of ints
        FD start and stop for plotting, None will select all FD
    fd_agg: str
        Aggregation of multiple predictions for a date, accepts 'min', 'max', 'mean'
    fd: int
        Specify FD to plot predictions vs actuals using only that FD
    average: bool
        If plotting average values or individual series
    series_name: str
        Series name (str) to plot
    top: bool
        Plot highest or lowest ordered series by mean target value
    data_subset: str
        Valid values are either holdout or allBacktests
    include_backtests: boolean (optional)
        Controls whether blender models are considered
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    acc_calc: function
        Function to calculate row-level prediction accuracy. Choose from mae, rmse, mape, smape, gamma, poission, and tweedie
        
    Returns:
    --------
    Plotly lineplot
    """
    
    assert isinstance(projects, list), 'Projects must be a list object'
    
    assert data_subset in [
            'allBacktests',
            'holdout',
            ], 'data_subset must be either allBacktests or holdout'

    assert fd_agg in [
            'min',
            'mean',
            'max',
            ], 'fd_agg accepts average, min, max, mean'
    
    date_col = ts_settings['date_col']
    series_id = ts_settings['series_id']
    target = ts_settings['target']
   
    print('Getting series predictions from DataRobot...')
    
    if series_name is None:
        series = _cut_series_by_rank(df, ts_settings, n=1, top=top)
        df_subset = df[df[series_id].isin(series)]
        series = "".join(df_subset[series_id].unique().tolist())
    
    if series_name is not None:
        series = series_name
        diff = len(set([series]).difference(set(df[series_id].unique().tolist())))
        assert diff == 0,\
        f'{series_id} {series} is not in the predictions file'
        df_subset = df[df[series_id].isin([series])]
    
    if average == True:
        df_subset = df
    
    df_subset = get_preds_and_actuals(
        df_subset,
        projects,
        ts_settings,
        n_models=1,
        data_subset=data_subset,
        include_blenders=include_blenders,
        metric=metric,
    )
    
    if fd_range is not None:
        assert len(fd_range) == 2, 'must provide two values for fd_range, ex: (1, 25)'
        assert fd_range[0] >= df_subset['forecast_distance'].min(), 'minumum forecast distance must be equal or greater to value in predictions'
        assert fd_range[1] <= df_subset['forecast_distance'].max(), 'maximum forecast distance be less than or equal to value in predictions'
        df_subset = df_subset[(df_subset['forecast_distance'] >= fd_range[0]) & (df_subset['forecast_distance'] <= fd_range[1])]

    if average == True:
        df_subset = df_subset.groupby([date_col])[[target, 'prediction']].mean().reset_index().drop_duplicates()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_subset[date_col], y=df_subset[target], line= dict(color='#EF553B', width= 2), name= 'Average Actuals'))
        fig.add_trace(go.Scatter(x=df_subset[date_col], y=df_subset['prediction'], line= dict(color='#636EFA', width= 2), name= 'Average Predictions'))
        fig.update_layout(title= f'Average \"{target}\" over forecast distance ')
        fig.show()
        
    else:
        if fd is not None:
            assert fd >= df_subset['forecast_distance'].min(), 'forecast distance to plot must be within prediction range, current value below minimum FD'
            assert fd <= df_subset['forecast_distance'].max(), 'forecast distance to plot must be within prediction range, current value above maximum FD'
            df_subset = df_subset[df_subset['forecast_distance'].astype(int) == fd]

        if fd_agg == 'min':
            df_subset = df_subset.groupby([date_col, series_id])[[target, 'prediction']].min().reset_index().drop_duplicates()

        elif fd_agg == 'max':
            df_subset = df_subset.groupby([date_col, series_id])[[target, 'prediction']].max().reset_index().drop_duplicates()

        else:
            df_subset = df_subset.groupby([date_col, series_id])[[target, 'prediction']].mean().reset_index().drop_duplicates()

        print('Plotting series actuals and predictions ...')

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_subset[date_col], y=df_subset[target],  line= dict(color='#EF553B', width= 2), legendgroup= 'Average Actuals', name = series + ' Actuals')),
        fig.add_trace(go.Scatter(x=df_subset[date_col], y=df_subset['prediction'],  line= dict(color='#636EFA', width= 2), legendgroup= 'Average Predictions', name = series +' Predictions'))
        
        if top is False:
            fig.update_layout(title_text='Bottom Series By Target Over Time')
        fig.update_layout(title_text='Top Series By Target Over Time')
        fig.update_layout(title= f'Individual series: {series} over forecast distance')
        fig.show()



def plot_series_acc(series_acc, ts_settings, data_subset='allBacktests', acc_calc=rmse, n=50):
    """
    Plots series-level and overall accuracy across multiple DataRobot projects

    cluster_acc: pandas df
        Output from get_series_acc()
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Choose from either holdout or allBacktests
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    Plotly barplot
    """
    n_series = len(series_acc[ts_settings['series_id']].unique())
    n = min(n_series, n)
    
    series_acc.sort_values(by=f'Series_{acc_calc.__name__.upper()}', ascending=False, inplace=True)

    series_acc = series_acc[0:n]

    fig = px.bar(
        series_acc,
        x=ts_settings['series_id'],
        y=f'Series_{acc_calc.__name__.upper()}',
        color='Cluster',
    ).for_each_trace(lambda t: t.update(name=t.name.replace('Project_Name=', '')))

    fig.add_trace(
        go.Scatter(
            x=series_acc[ts_settings['series_id']],
            y=series_acc[f'Total_{acc_calc.__name__.upper()}'],
            mode='lines',
            marker=dict(color='black'),
            name=f'Overall {acc_calc.__name__.upper()}',
        )
    )

    fig.update_yaxes(title=acc_calc.__name__.upper())
    fig.update_xaxes(tickangle=45)
    fig.update_layout(title_text=f'Series Accuracy - {data_subset}')
    fig.show()


def get_project_info(df):
    """
    Parse project name to get FD, FDW, and Cluster information

    Returns:
    --------
    pandas df
    """
    df = df.copy()
    try:
        df['Cluster'] = df['Project_Name'].apply(lambda x: x.split('_Cluster-')).apply(lambda x: x[1])
        df['FD'] = df['Project_Name'].apply(lambda x: x.split('_FD:')[1].split('_FDW:')[0])
        df['FDW'] = df['Project_Name'].apply(lambda x: x.split('_FDW:')[1].split('_Cluster-')[0])
    except:
        df['Cluster'] = 'all_series'
        df['FD'] = df['Project_Name'].apply(lambda x: x.split('_FD:')[1].split('_FDW:')[0])
        df['FDW'] = df['Project_Name'].apply(lambda x: x.split('_FDW:')[1].split('all_series')[0])

    # df['FD'] = df['Project_Name'].apply(lambda x: x.split('_FD:')[1].split('_FDW:')[0])
    # df['FDW'] = df['Project_Name'].apply(lambda x: x.split('_FDW:')[1].split('_Cluster-')[0])

    return df


def filter_best_fdw_scores(scores, col_error='All_Backtests_RMSE'):
    """
    Subset df to projects with the best error metric for each FD and Cluster pair

    scores: pandas df
        Output from get_or_request_backtest_scores()
    col_error: str
        Column name from scores df

    Returns:
    --------
    pandas df
    """
    df = get_project_info(scores)
    df['_tmp'] = df[col_error].apply(lambda x: np.nanmean(np.array(x, dtype=np.float32)))
    idx = df.groupby(['Cluster', 'FD']).apply(lambda x: x['_tmp'].idxmin()).values
    return scores.iloc[idx, :]


def filter_best_fdw_projects(scores, projects, col_error='All_Backtests_RMSE'):
    """
    Subset list to projects with the best error metric for each FD and Cluster pair

    scores: pandas df
        Output from get_or_request_backtest_scores()
    projects: list
        DataRobot projects object(s)
    col_error: str
        Column name from scores df

    Returns:
    --------
    list

    """
    df = filter_best_fdw_scores(scores, col_error)
    return [p for p in projects if p.project_name in df['Project_Name'].unique()]


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



def plot_fd_accuracy(df, projects, ts_settings, data_subset='allBacktests', metric='RMSE'):
    """
    Plots accuracy over forecast distance

    df: pandas df
        Input data
    projects: list
        List of DataRobot datetime projects
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Choose from either holdout or allBacktests
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'

    Returns:
    --------
    Plotly lineplot
    """
    
    assert isinstance(projects, list), 'Projects must be a list object'
    assert data_subset in [
        'allBacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    mapper = {
        'MAE': mae,
        'SMAPE': smape,
        'MAPE': mape,
        'RMSE': rmse,
        'Gamma Deviance': gamma_loss,
        'Tweedie Deviance': tweedie_loss,
        'Poisson Deviance': poisson_loss,
    }

    df = get_preds_and_actuals(
        df, projects, ts_settings, n_models=1, data_subset=data_subset, metric=metric
    )
    df = (
        df.groupby(['Project_Name', 'forecast_distance'])
        .apply(lambda x: mapper[metric](x[ts_settings['target']], x['prediction']))
        .reset_index()
    )

    df.columns = ['Project_Name', 'forecast_distance', mapper[metric].__name__.upper()]
    fig = px.line(
        df, x='forecast_distance', y=mapper[metric].__name__.upper(), color='Project_Name'
    ).for_each_trace(lambda t: t.update(name=t.name.replace('Project_Name=', '')))

    fig.update_layout(title_text='Forecasting Accuracy per Forecast Distance')
    fig.update_yaxes(title=mapper[metric].__name__.upper())
    fig.update_xaxes(title='Forecast Distance')
    fig.show()


def plot_fd_accuracy_by_cluster(
    df, scores, projects, ts_settings, data_subset='holdout', metric='RMSE', split_col='Cluster'
):
    """
    Plots accuracy over forecast distance by cluster

    df: pandas df
        Input data
    scores: pandas df
        Output from get_or_request_backtest_scores()
    projects: list
        List of DataRobot datetime projects
    ts_settings: dict
        Parameters for time series project
    data_subset: str (optional)
        Choose from either holdout or allBacktests
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
        Choose from list of 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    split_col: str (optional)
        Column name to be used to split by cluster

    Returns:
    --------
    Plotly lineplot
    """
    scores = get_project_info(scores)

    for c in scores[split_col].unique():
        project_names = list(
            scores.loc[scores[split_col] == c, 'Project_Name'].reset_index(drop=True)
        )
        projects_by_cluster = [p for p in projects if p.project_name in project_names]
        plot_fd_accuracy(df, projects_by_cluster, ts_settings, data_subset, metric)


###########################
# Performance Improvements
###########################


def get_reduced_features_featurelist(project, model, threshold=0.99):
    """
    Helper function for train_reduced_features_models()

    project: DataRobot project object
    model: DataRobot model object
    threshold: np.float

    Returns:
    --------
    DataRobot featurelist
    """
    print(
        f'Collecting Feature Impact for M{model.model_number} in project {project.project_name}...'
    )

    impact = pd.DataFrame.from_records(model.get_or_request_feature_impact())
    impact['impactUnnormalized'] = np.where(
        impact['impactUnnormalized'] < 0, 0, impact['impactUnnormalized']
    )
    impact['cumulative_impact'] = (
        impact['impactUnnormalized'].cumsum() / impact['impactUnnormalized'].sum()
    )

    to_keep = np.where(impact['cumulative_impact'] <= threshold)[0]
    if len(to_keep) < 1:
        print('Applying this threshold would result in a featurelist with no features')
        return None

    idx = np.max(to_keep)

    selected_features = impact.loc[0:idx, 'featureName'].to_list()
    feature_list = project.create_modeling_featurelist(
        f'Top {len(selected_features)} features M{model.model_number}', selected_features
    )

    return feature_list

    
def train_reduced_features_models(
    projects,
    n_models=1,
    threshold=0.99,
    data_subset='allBacktests',
    include_blenders=True,
    metric=None,
    iteration=False,
    model_rank= None,
    model_id = None
):
    """
    Retrain top models with reduced feature featurelists

    projects: list
        DataRobot project object(s)
    n_models: int
        Number of models to retrain with reduced feature featurelists
    threshold: np.float
        Controls the number of features to keep in the reduced feature list. Percentage of cumulative feature impact
    data_subset: str
        Choose from either holdout or allBacktests
    include_blenders: boolean (optional)
        Include blender models
    metric: str (optional)
        Project metric used to sort the DataRobot leaderboard
    iteration: boolean (optional)
        Optional parameter used to output length of feature list for some functions
    model_rank: int (optional)
        None if top model, model leaderboard rank if any model other than top desired
    model_id: str (optional)
        DataRobot model id

    Returns:
    --------
    (optional) Pandas df
    
    """
    
    assert isinstance(projects, list), 'Projects must be a list object'

    scores = pd.DataFrame([])

    for p in projects:
        if model_id is not None:
            models = [dr.Model.get(project=p.id, model_id= model_id)]
        elif model_rank:
            models = get_ranked_model(p, model_rank, metric= None, data_subset= 'allBacktests')
        else:
            models = get_top_models_from_project(p, n_models, data_subset, include_blenders, metric)

        for m in models:
            try:
                feature_list = get_reduced_features_featurelist(p, m, threshold)

                if feature_list is None:
                    continue
                try:
                    m.train_datetime(featurelist_id=feature_list.id)
                    print(f'Training {m.model_type} on Featurelist {feature_list.name}') # m.id))
                except dr.errors.ClientError as e:
                    print(e)
                    if iteration and "This blueprint requires" in str(e):
                        print('***** WARNING: This model may not support retraining on the smaller feature list. Your learning curve may be truncated *****')

            except dr.errors.ClientError as e:
                print(e)
  
        if iteration:
            tmp = get_or_request_model_scores(p, m, data_subset, metric= metric)

            scores = scores.append(tmp)

    if iteration:
        return scores
                

def test_feature_selection(df, 
                            projects, 
                            ts_settings,
                            n_models= 1,
                            model_id= None,
                            data_subset='allBacktests', 
                            metric='RMSE', 
                            threshold_range= (0.6, 1.0), 
                            step_size= 0.1,
                            model_rank= None):
    '''
    Perform automated, iterative feature selection through a range of feature importance thresholds

    df: pandas df
    projects: list
        list of DataRobot projects for feature list selection
    ts_settings: dict
        Parameters for time series project
    n_models: int
        number of models to generate feature lists from
    model_id: str
        DataRobot model id
    data_subset: str
        Choose from either holdout or allBacktests
    metric: str
        Metric to be used for sorting the leaderboard, if None uses project metric
    threshold_range: tuple of np.floats (optional)
        upper and lower bounds of threshold for feature selection, percentage of cumulate feature impact
    step_size: np.float (optional)
        step-size across threshold-range
    model_rank: int (optional)
        None if top model, model leaderboard rank if any model other than top desired
    --------
    Returns:
    Pandas DataFrame
    '''
    
    assert step_size >= 0.05, 'Minimum threshold step-size is 0.05'

    results = pd.DataFrame()

    for step in np.arange(threshold_range[0], threshold_range[1], step_size)[::-1]:
        step = float("%.2f" % step)
        # train a model with a reduced set of features
        info = train_reduced_features_models(projects, threshold= step, include_blenders= False, metric= metric, iteration= True, model_rank= model_rank, model_id = model_id)
        results = results.append(info)

    # add the length of each feature list and model id
    results.drop_duplicates(subset= ['Project_ID','Model_ID'], inplace=True)

    return results


def run_feature_selection_projects(df,
                                   projects,
                                   ts_settings,
                                   data_subset='allBacktests',
                                   metric=None,
                                   threshold_range=(0.6, 1.0),
                                   step_size=0.1,
                                   plot= False):
    '''
        Perform automated, iterative feature selection through a range of feature importance thresholds for many projects, automatically selecting the best non-blender model that can be retrained

    df: pandas df
    projects: list
        list of DataRobot projects for feature list selection
    ts_settings: dict
        Parameters for time series project
    data_subset: str
        Choose from either holdout or allBacktests
    metric: str
        Metric to be used for sorting the leaderboard, if None uses project metric
    threshold_range: tuple of np.floats (optional)
        upper and lower bounds of threshold for feature selection, percentage of cumulate feature impact
    step_size: np.float (optional)
        step-size across threshold-range
    plot: bool (optional)
        Plot individual featurelist learning curves for all projects
    --------
    Returns:
    Pandas DataFrame
    '''

    assert step_size >= 0.05, 'Minimum threshold step-size is 0.05'

    results = pd.DataFrame()
    project_ranks = []
    models = []
    print(f'Getting ranked models from {len(projects)} projects ...')
    for project in projects:
        for i in range(1, 11):
            model = get_ranked_model(project, model_rank=i, metric=None, data_subset='allBacktests')
            if not any(x in str(model[0]) for x in ['Blender', 'Zero', 'Baseline']):
                project_ranks.append((project, i, model[0].id))
                models.append(model[0])
                break
            if i == 10:
                print(f'{project.project_name} top-10 models may not support retraining on reduced features')
                project_ranks.append((project, 1, model[0].id))
                models.append(model[0])

    print(f'Training reduced feature lists for {len(projects)} projects ...')
    # project_ranks = [x for x in project_ranks if x[1] != 1]
    for project, rank_num, id in project_ranks:
        print(f'\nRetraining the {rank_num}-ranked model ...')
        print("------------")
        data = test_feature_selection(df, [project], ts_settings, model_id=id,
                                      threshold_range=threshold_range, step_size=step_size)
        if plot:
            plot_featurelist_learning_curve(data, data_subset='allBacktests', metric='RMSE')
        results = results.append(data)
    
    # score backtests on models
    for m in models:
        try:
            print(type(m))
            m.score_backtests()
        except:
            print(f'Could not score {m}')
    print(f'Scoring backtests for {len(models)} models retrained with reduced features...')
        
    return results # .drop_duplicates()


def plot_featurelist_learning_curve(df, data_subset='allBacktests', metric= None):
    """
    Plot the featurelist length and error metric to generate a learning curve

    df: Pandas df
        Contains information on feature lists, and accuracy for iterations on a model. output of test_feature_selection()
    data_subset: str
        desired backtest to plot. Inputs are: 'backtest_1, all_Backtests, holdout'
    metric: str
        error metric to plot. Inputs are: 'RMSE', 'MASE', 'Theils_U', 'SMAPE', 'R_Squared'
    
    Returns:
    --------
    Plotly lineplot
    """
    
    assert data_subset.lower() in [
        'backtest_1',
        'allbacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'
    assert metric.upper() in [
        'RMSE', 
        'MASE', 
        'Theils_U', 
        'SMAPE', 
        'R_Squared'
    ], "metric must be 'RMSE','MASE', 'Theils_U', 'SMAPE', 'R_Squared'"
    
    df = df[df['Featurelist'].str.contains('(?<=Top )\d*(?= features)', regex= True)].copy()
    df['Feature_length'] = df['Featurelist'].apply(lambda x: re.search('(?<=Top )\d*(?= features)', x).group(0))
    
    if data_subset == 'allBacktests':
        data_subset = data_subset.title().replace('b','_B')
        metric_column = data_subset + "_" + metric
        df[metric_column] = df[metric_column].apply(lambda x: mean([v for v in x if v != None]))
        df = df[['Feature_length', metric_column]].drop_duplicates()
    else:
        data_subset = data_subset.title()
        metric_column = data_subset + "_" + metric
        df = df[['Feature_length', metric_column]].drop_duplicates()

    df.drop_duplicates(inplace= True)

    df = df[['Feature_length', metric_column]].sort_values('Feature_length', ascending= True)
    fig = px.scatter(df, x='Feature_length', y=metric_column)

    fig.update_layout(title_text='Top Series By Target Over Time')
    fig.show()


def plot_all_featurelist_curves(df, ts_settings, data_subset='allBacktests', metric='RMSE'):
    """
    Plot all reduced featurelists curves on the same plot

    df: pandas df
    ts_settings: dict
        Parameters for DR datetime projects
    data_subset: str
        data to be used for plotting
    metric: str
        metric used for plotting

    Returns:
    --------
    Plotly lineplot
    """

    assert data_subset.lower() in [
        'backtest_1',
        'allbacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'
    assert metric in [
        'RMSE',
        'MASE',
        'Theils_U',
        'SMAPE',
        'R_Squared'
    ], "metric must be 'RMSE', 'MASE', 'Theils_U', 'SMAPE', 'R_Squared'"

    df = df[df['Featurelist'].str.contains('(?<=Top )\d*(?= features)', regex=True)].copy()
    df['Featurelist_length'] = df['Featurelist'].apply(
        lambda x: int(re.search('(?<=Top )\d*(?= features)', x).group(0)))

    if data_subset == 'allBacktests':
        data_subset = data_subset.capitalize().replace('b', '_B')
        metric_column = data_subset + "_" + metric
        df[metric_column] = df[metric_column].apply(lambda x: np.mean([(float(v)) for v in x if v != None]))
        df = df[['Project_Name', 'Featurelist_length', metric_column]].drop_duplicates().sort_values(
            ['Featurelist_length', 'Project_Name'], ascending=[False, True])  
    else:
        data_subset = data_subset.capitalize()
        metric_column = data_subset + "_" + metric
        df = df[['Project_Name', 'Featurelist_length', metric_column]].drop_duplicates().sort_values(
            ['Featurelist_length', 'Project_Name'], ascending=[False, True])  

    print(metric_column)
    num = df['Project_Name'].nunique()
    fig = px.line(df, x='Featurelist_length', y=metric_column, color='Project_Name')
    fig.update_layout(title_text=f'Feature List Selection Curves for {num} Projects')
    fig.update_layout(yaxis=dict(range=[min(df[metric_column].values)* 0.8,max(df[metric_column].values)* 1.1]))
    fig.update_layout(xaxis=dict(range=[0, max(df['Featurelist_length'].values)+1]))
    fig.show()
