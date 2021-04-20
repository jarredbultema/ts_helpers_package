import datetime as dt
import time
from collections import Counter
import datarobot as dr
import numpy as np
import pandas as pd
from tqdm import tqdm

from .ts_data_quality import get_timestep
from .ts_projects import *


###################
# Project Creation
###################


def create_dr_project(df, project_name, ts_settings, advanced_options= {'weights': None}):
    """
    Kickoff single DataRobot project

    df: pandas df
    project_name: name of project
    ts_settings: dictionary of parameters for time series project

    Returns:
    --------
    DataRobot project object

    """

    # print(f'Building Next Project \n...\n')

    #######################
    # Get Advanced Options
    #######################
    # opts = {
    #     'weights': None,
    #     'response_cap': None,
    #     'blueprint_threshold': None,
    #     'seed': None,
    #     'smart_downsampled': False,
    #     'majority_downsampling_rate': None,
    #     'offset': None,
    #     'exposure': None,
    #     'accuracy_optimized_mb': None,
    #     'scaleout_modeling_mode': None,
    #     'events_count': None,
    #     'monotonic_increasing_featurelist_id': None,
    #     'monotonic_decreasing_featurelist_id': None,
    #     'only_include_monotonic_blueprints': None,
    # }
    #
    # for opt in advanced_options.items():
    #     opts[opt[0]] = opt[1]
    #
    # opts = dr.AdvancedOptions(
    #     weights=opts['weights'],
    #     seed=opts['seed'],
    #     monotonic_increasing_featurelist_id=opts['monotonic_increasing_featurelist_id'],
    #     monotonic_decreasing_featurelist_id=opts['monotonic_decreasing_featurelist_id'],
    #     only_include_monotonic_blueprints=opts['only_include_monotonic_blueprints'],
    #     accuracy_optimized_mb=opts['accuracy_optimized_mb'],
    #     smart_downsampled=opts['smart_downsampled'],
    # )

    # simple should work, but require at least 1 'dummy' default argument for the **advanced_options kwarg
    opts = dr.AdvancedOptions(**advanced_options)

    ############################
    # Get Datetime Specification
    ############################
    settings = {
        'max_date': None,
        'known_in_advance': None,
        'do_not_derive': None,
        'num_backtests': None,
        'validation_duration': None,
        'holdout_duration': None,
        'holdout_start_date': None,
        'disable_holdout': False,
        'number_of_backtests': None,
        'backtests': None,
        'use_cross_series_features': None,
        'aggregation_type': None,
        'cross_series_group_by_columns': None,
        'calendar_id': None,
        'use_time_series': False,
        'series_id': None,
        'metric': None,
        'target': None,
        'mode': dr.AUTOPILOT_MODE.FULL_AUTO,  # MANUAL #QUICK
        'date_col': None,
        'fd_start': None,
        'fd_end': None,
        'fdw_start': None,
        'fdw_end': None,
    }

    for s in ts_settings.items():
        settings[s[0]] = s[1]

    df[settings['date_col']] = pd.to_datetime(df[settings['date_col']])

    if settings['max_date'] is None:
        settings['max_date'] = df[settings['date_col']].max()
    else:
        settings['max_date'] = pd.to_datetime(settings['max_date'])

    if ts_settings['known_in_advance'] is not None:
        settings['known_in_advance'] = [
            dr.FeatureSettings(feat_name, known_in_advance=True)
            for feat_name in ts_settings['known_in_advance']
        ]
    
    if ts_settings['do_not_derive'] is not None:
        settings['do_not_derive'] = [
            dr.FeatureSettings(feat_name, do_not_derive=True)
            for feat_name in ts_settings['do_not_derive']
        ]

    # create the appropriate feature settings list for project configuration
    if all(v is not None for v in [ts_settings['known_in_advance'], ts_settings['do_not_derive']]):
        combined_feature_settings = settings['known_in_advance'] + settings['do_not_derive']
    elif ts_settings['known_in_advance'] is not None:
        combined_feature_settings = settings['known_in_advance']
    elif ts_settings['do_not_derive'] is not None:
        combined_feature_settings = settings['do_not_derive']
    else:
        combined_feature_settings = None

    # Update validation and holdout duration, start, and end date
    project_time_unit, project_time_step = get_timestep(df, settings)

    validation_durations = {'minute': 0, 'hour': 0, 'day': 0, 'month': 0}
    holdout_durations = {'minute': 0, 'hour': 0, 'day': 0, 'month': 0}

    if project_time_unit == 'minute':
        validation_durations['minute'] = settings['validation_duration']
        holdout_durations['minute'] = settings['holdout_duration']

    elif project_time_unit == 'hour':
        validation_durations['hour'] = settings['validation_duration']
        holdout_durations['hour'] = settings['holdout_duration']

    elif project_time_unit == 'day':
        validation_durations['day'] = settings['validation_duration']
        holdout_durations['day'] = settings['holdout_duration']

    elif project_time_unit == 'week':
        validation_durations['day'] = settings['validation_duration'] * 7
        holdout_durations['day'] = settings['holdout_duration'] * 7

    elif project_time_unit == 'month':
        validation_durations['day'] = settings['validation_duration'] * 31
        holdout_durations['day'] = settings['holdout_duration'] * 31

    else:
        raise ValueError(f'{project_time_unit} is not a supported timestep')

    if settings['disable_holdout']:
        settings['holdout_duration'] = None
        settings['holdout_start_date'] = None
    else:
        settings['holdout_start_date'] = settings['max_date'] - dt.timedelta(
            minutes=holdout_durations['minute'],
            hours=holdout_durations['hour'],
            days=holdout_durations['day'],
        )

        settings['holdout_duration'] = dr.partitioning_methods.construct_duration_string(
            minutes=holdout_durations['minute'],
            hours=holdout_durations['hour'],
            days=holdout_durations['day'],
        )

    ###############################
    # Create Datetime Specification
    ###############################
    time_partition = dr.DatetimePartitioningSpecification(
        feature_settings= combined_feature_settings,
        # gap_duration = dr.partitioning_methods.construct_duration_string(years=0, months=0, days=0),
        validation_duration=dr.partitioning_methods.construct_duration_string(
            minutes=validation_durations['minute'],
            hours=validation_durations['hour'],
            days=validation_durations['day'],
        ),
        datetime_partition_column=settings['date_col'],
        use_time_series=settings['use_time_series'],
        disable_holdout=settings['disable_holdout'],  # set this if disable_holdout is set to False
        holdout_start_date=settings['holdout_start_date'],
        holdout_duration=settings[
            'holdout_duration'
        ],  # set this if disable_holdout is set to False
        multiseries_id_columns=[settings['series_id']],
        forecast_window_start=int(settings['fd_start']),
        forecast_window_end=int(settings['fd_end']),
        feature_derivation_window_start=int(settings['fdw_start']),
        feature_derivation_window_end=int(settings['fdw_end']),
        number_of_backtests=settings['num_backtests'],
        calendar_id=settings['calendar_id'],
        use_cross_series_features=settings['use_cross_series_features'],
        aggregation_type=settings['aggregation_type'],
        cross_series_group_by_columns=settings['cross_series_group_by_columns'],
    )

    ################
    # Create Project
    ################
    project = dr.Project.create(
        project_name=project_name, sourcedata=df, max_wait=14400, read_timeout=14400
    )

    # print(f'Creating project {project_name} ...')

    #################
    # Start Autopilot
    #################
    project.set_target(
        target=settings['target'],
        metric=settings['metric'],
        mode=settings['mode'],
        advanced_options=opts,
        worker_count=-1,
        partitioning_method=time_partition,
        max_wait=14400,
    )

    return project


def create_dr_projects(
    df, ts_settings, prefix='TS', split_col=None, fdws=None, fds=None, advanced_options= {'weights': None}):
    """
    Kickoff multiple DataRobot projects

    df: pandas df
    ts_settings: dictionary of parameters for time series project
    prefix: str to concatenate to start of project name
    split_col: column in df that identifies cluster labels
    fdws: list of tuples containing feature derivation window start and end values
    fds: list of tuples containing forecast distance start and end values

    Returns:
    --------
    List of projects

    Example:
    --------
    split_col = 'Cluster'
    fdws=[(-14,0),(-28,0),(-62,0)]
    fds = [(1,7),(8,14)]
    """

    if fdws is None:
        fdws = [(ts_settings['fdw_start'], ts_settings['fdw_end'])]

    if fds is None:
        fds = [(ts_settings['fd_start'], ts_settings['fd_end'])]

    clusters = range(1) if split_col is None else df[split_col].unique()

    assert isinstance(fdws, list), 'fdws must be a list object'
    assert isinstance(fds, list), 'fds must be a list object'
    if split_col:
        assert len(df[split_col].unique()) > 1, 'There must be at least 2 clusters'

    n_projects = len(clusters) * len(fdws) * len(fds)
    projects = []
    failed_projects = []

    with tqdm(range(n_projects), position=0, leave=True, desc= f'Building {n_projects} projects') as pbar: # progress bars
        for c in clusters:
            for fdw in fdws:
                for fd in fds:

                    ts_settings['fd_start'], ts_settings['fd_end'] = fd[0], fd[1]
                    ts_settings['fdw_start'], ts_settings['fdw_end'] = fdw[0], fdw[1]
                    cluster_suffix = 'all_series' if split_col is None else 'Cluster-' + str(c) #.astype('str')

                    # Name project
                    project_name = '{prefix}_FD:{start}-{end}_FDW:{fdw}_{cluster}'.format(
                        prefix=prefix,
                        fdw=ts_settings['fdw_start'],
                        start=ts_settings['fd_start'],
                        end=ts_settings['fd_end'],
                        cluster=cluster_suffix,
                    )

                    if split_col is not None:
                        data = df.loc[df[split_col] == c, :].copy()
                        data.drop(columns=split_col, axis=1, inplace=True)
                    else:
                        data = df.copy()

                    # Create project
                    # updated method with error-handling
                    try:
                        project = create_dr_project(
                            data, project_name, ts_settings, advanced_options=advanced_options
                        )
                        projects.append(project)
                        pbar.set_postfix_str(f'Project {project_name} was successfully built!', refresh= False)
                        pbar.update()

                    except Exception as ex:
                        failed_projects.append(project_name)
                        pbar.set_postfix_str(f'Something went wrong during project creation for {project_name}: {ex}', refresh= False)
                        pbar.update()
    pbar.close()
    if len(failed_projects) > 0:
        print(f'***** The following {len(failed_projects)} projects were not built: {failed_projects} *****')
    else:
        print(f'***** All {len(projects)} projects were successfully built! *****\n')
    return projects


def wait_for_jobs_to_process(projects):
    """
    Check if any DataRobot jobs are still processing
    
    projects: list
        list of DataRobot project object
    """
    
    all_jobs = np.sum([len(p.get_all_jobs()) for p in projects])
    while all_jobs > 0:
        print(f'There are {all_jobs} jobs still processing')
        time.sleep(60)
        all_jobs = np.sum([len(p.get_all_jobs()) for p in projects])

    print('All jobs have finished processing...')


def train_timeseries_blender(project, models, n_models=None, blender_method='AVERAGE', data_subset='allBacktests'):
    '''
    Train timeseries blenders for a DataRobot Datetimemodels

    project: DataRobot project object
        DataRobot project in which to create blenders
    models: list (optional)
        DataRobot Datetimemodel model ids
    n_models: int (optional)
        Use top n_models to create blenders
    blender_method: str
        Type of blender to create
    data_subset: str
        desired backtest to get top models. Inputs are: 'backtest_1, all_Backtests, holdout'
    '''
    
    from .ts_projects import get_top_models_from_project, get_top_models_from_projects
    
    assert isinstance(models, list) or models is None, 'models must be a list or Nonetype'
    assert isinstance(n_models, int) or n_models is None, 'n_models must be a list or NoneType'
    assert blender_method in [
        'FORECAST_DISTANCE_AVG',
        'AVERAGE',
        'MEDIAN',
        'FORECAST_DISTANCE_ENET'
    ], 'blender_method must be FORECAST_DISTANCE_AVG, AVERAGE, MEDIAN, FORECAST_DISTANCE_ENET'
    assert data_subset.lower() in [
        'backtest_1',
        'allbacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    if models is not None:
        info = [(x, dr.Model.get(project=project.id, model_id=x).training_duration) for x in models]
        durations = [x[1] for x in info]
        models = [x[0] for x in info if x[1] == max(durations, key=Counter(durations).get)]

    if n_models is not None:
        info = [(x.id, x.training_duration) for x in
                get_top_models_from_project(project, n_models=n_models, data_subset=data_subset,
                                            include_blenders=False)]
        durations = [x[1] for x in info]
        models = [x[0] for x in info if x[1] == max(durations, key=Counter(durations).get)]

    if blender_method == 'MEDIAN':
        method = dr.enums.BLENDER_METHOD.MEDIAN
    elif blender_method == 'FORECAST_DISTANCE_AVG':
        method = dr.enums.BLENDER_METHOD.FORECAST_DISTANCE_AVG
    elif blender_method == 'FORECAST_DISTANCE_ENET':
        method = dr.enums.BLENDER_METHOD.FORECAST_DISTANCE_ENET
    else:
        method = dr.enums.BLENDER_METHOD.AVERAGE

    print(
        f'Creating {blender_method.replace("_", " ").title()} blender using {len(models)} models from {project.project_name} ... ')
    project.blend(models, method)


def train_timeseries_blender_projects(projects, models, n_models=None, blender_method='AVERAGE',
                                      data_subset='allBacktests'):
    '''
    Train timeseries blenders for multiple DataRobot projects

    projects: list
        DataRobot project objects in which to create blenders
    models: list of lists (optional)
        list of DataRobot Datetimemodel model ids for each project
    n_models: int (optional)
        Use top n_models to create blenders
    blender_method: str
        Type of blender to create
    data_subset: str
        desired backtest to get top models. Inputs are: 'backtest_1, all_Backtests, holdout'
    '''

    assert isinstance(projects, list), 'projects argument must be a list'
    assert isinstance(models, list) or models is None, 'models must be a list or Nonetype'
    assert isinstance(n_models, int) or n_models is None, 'n_models must be a list or NoneType'
    assert blender_method in [
        'FORECAST_DISTANCE_AVG',
        'AVERAGE',
        'MEDIAN',
        'FORECAST_DISTANCE_ENET'
    ], 'blender_method must be FORECAST_DISTANCE_AVG, AVERAGE, MEDIAN, FORECAST_DISTANCE_ENET'
    assert data_subset.lower() in [
        'backtest_1',
        'allbacktests',
        'holdout',
    ], 'data_subset must be either allBacktests or holdout'

    if models is not None:
        for idx, p in enumerate(projects):
            try:
                train_timeseries_blender(p, models=models[idx], n_models=n_models, blender_method=blender_method,
                                         data_subset=data_subset)
            except dr.errors.ClientError as e:
                print(e)

    if n_models is not None:
        for p in projects:
            try:
                train_timeseries_blender(p, models=models, n_models=n_models, blender_method=blender_method)
            except dr.errors.ClientError as e:
                print(e)


def run_repository_models(projects, n_bps=None, insane=False, exclude=['Mean', 'Eureqa', 'Keras', 'VARMAX']):
    """
    Run blueprints from the repository using the feature list from the DataRobot recommended models

    projects: list
        DataRobot project object(s)
    n_bps: int
        Number of blueprints from repository to return
    insane: bool
        If True, run repo on featurelist from top 5 blueprints on leaderboard, if False run on recommended model featurelist
    exclude: list
        DataRobot model types to exclude from running
    """

    from .ts_projects import get_top_models_from_project, get_top_models_from_projects
    
    assert isinstance(projects, list), 'projects must be a list object'
    if n_bps is not None:
        assert isinstance(n_bps, int), 'n_bps must be an integer'
        assert n_bps > 0, 'n_bps must be larger than 0'

    for p in projects:
        recommended_models = [dr.Model.get(project=p.id, model_id=x.model_id) for x in
                              dr.ModelRecommendation.get_all(p.id)]
        recommended_featurelists = [x.featurelist_id for x in recommended_models if
                                    'Multiple' not in str(x.featurelist_name)]
        training_duration = [x for x in recommended_models if 'Deployment' not in str(x)][0].training_duration
        bps = [bp for bp in p.get_blueprints() if all([f not in bp.model_type for f in exclude])]

        if n_bps is None:
            n_bps = len(bps)

        if insane:
            recommended_models += [dr.Model.get(project=p.id, model_id=x.id) for x in
                                   get_top_models_from_project(p, n_models=5, include_blenders=False)]
            recommended_featurelists += [x.featurelist_id for x in recommended_models if
                                         'Multiple' not in str(x.featurelist_name)]
            recommended_featurelists = list(set(recommended_featurelists))

        bps = bps[0:n_bps]
        bps = [(bp.id, fl) for bp in bps for fl in recommended_featurelists]
        good = 0
        bad = 0

        print(f'Attemping training of {len(bps)} blueprints in {p.project_name} ...')
        for idx, bp in enumerate(bps):
            try:
                p.train_datetime(
                    blueprint_id=bp[0], featurelist_id=bp[1], training_duration=training_duration
                )
                good += 1
            except dr.errors.ClientError or dr.errors.JobAlreadyRequested as e:  # .JobAlreadyRequested
                bad += 1
        print(f'{good} new blueprints were trained')
        print(f'{bad} blueprints were unable to train or had an error\n')


def retrain_to_forecast_point(
        project,
        ts_settings,
        forecast_point,
        model=None,
        duration=None,
        metric=None
):
    '''
    Retrains a frozen model up to a specified forecast point

    Inputs:
    -------
    project: project object
        DataRobot project
    ts_settings: dict
        Settings for DataRobot time series project
    forecast_point: str or datetime object
        Date to be used as a reference for end of training and start of predictions
    model: DataRobot datetime model or None (optional)
        Model to be retrained, if None, top non-blender will be used
    duration: int or None (optional)
        Duration of training data to be used, If None uses project settings to determine duration
    metric: str (optional)
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Output:
    -------
    retrained model object
    '''

    #     assert isinstance(project, dr.models.project.Project), 'project must be a DataRobot project object'
    #     assert isinstance(model, dr.models.model.DatetimeModel) or isinstance(model, NoneType), 'model must be a DataRobot datetime model object or None'
    #     assert isinstance(forecast_point, str or datetime.date), 'forecast_point must be a str or datetime.date'

    from .ts_projects import get_top_models_from_project, get_top_models_from_projects
    
    if model is None:
        model = get_top_models_from_project(project=project, n_models=2, include_blenders=False, metric=metric)
    if 'Baseline' in str(model[0]):
        print('Top model is a "Baseline" model, attempting to use the 2nd highest ranked model')
        model = model[1]
    else:
        model = model[0]
    #     assert 'Baseline' not in str(model), 'Top model is a "Baseline" model, select a different metric for sorting or specify a model'

    if duration is None:
        duration = int("".join(re.findall('\d', model.training_info['prediction_training_duration'])))

    forecast_point = pd.to_datetime(forecast_point)
    start = forecast_point - pd.DateOffset(duration)

    print(f'Retraing Frozen {model} from {start.date()} to {forecast_point.date()}... ')
    retrain_model = model.request_frozen_datetime_model(training_start_date=start, training_end_date=forecast_point)
    wait_for_jobs_to_process([project])
    print('Retraining Complete!')
    m_id = dr.ModelJob.get_model(project_id=project.id,
                                 model_job_id=retrain_model.id).id
    return [x for x in project.get_datetime_models() if x.id == m_id][0]


def retrain_projects_to_forecast_point(
        projects,
        ts_settings,
        forecast_point,
        models=None,
        duration=None,
        metric=None
):
    '''
    Retrains projects on a frozen model up to a specified forecast point

    Inputs:
    -------
    projects: list
        List of DataRobot projects
    ts_settings: dict
        Settings for DataRobot time series project
    forecast_point: str or datetime object
        Date to be used as a reference for end of training and start of predictions
    models: list (optional)
        List of DataRobot datetime models to be retrained. If None, top non-blender model will be used
    duration: int or None (optional)
        Duration of training data to be used, If None uses project settings to determine duration
    metric: str (optional)
        Choose from list of 'MASE', 'RMSE', 'MAPE', 'SMAPE', 'MAE', 'R Squared', 'Gamma Deviance',
                            'SMAPE', 'Tweedie Deviance', 'Poisson Deviance', or 'RMSLE'
    Output:
    -------
    retrained_models: list of DataRobot model objects
    '''
    retrained_models = []
    for idx, p in enumerate(projects):
        print(f'For Project {p}:')
        m = models
        if models is not None:
            m = models[idx]
        rt_model = retrain_to_forecast_point(p, ts_settings, forecast_point=forecast_point, model=m, duration=duration,
                                             metric=metric)
        retrained_models.append(rt_model)
    return retrained_models