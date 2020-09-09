import datetime as dt

import datarobot as dr
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    Holiday,
    TH,
    USFederalHolidayCalendar,
)
import plotly.express as px

from .ts_data_quality import get_timestep


#####################
# Calendar Functions
#####################


def create_ts_calendar(df, ts_settings, additional_events=None):
    """
    df: pandas df
    ts_settings: dict
        Parameters for time series project
    additional_events: pandas df(optional)
        df of additional events to add to calendar

    Returns:
    --------
    Calendar of events

    """
    date_col = ts_settings['date_col']

    cal = USFederalHolidayCalendar(AbstractHolidayCalendar)

    black_friday = Holiday(
        "Black Friday", month=11, day=1, offset=[pd.DateOffset(weekday=TH(4)), pd.DateOffset(1)]
    )
    cyber_monday = Holiday(
        "Cyber Monday", month=11, day=1, offset=[pd.DateOffset(weekday=TH(4)), pd.DateOffset(4)]
    )
    cal.rules.append(black_friday)
    cal.rules.append(cyber_monday)

    cal.rules[9] = Holiday('Christmas', month=12, day=25)
    cal.rules[4] = Holiday('July 4th', month=7, day=4)

    events = pd.DataFrame(
        cal.holidays(
            start=pd.to_datetime(df[date_col].min()),
            end=pd.to_datetime(df[date_col].max()) + dt.timedelta(days=365),
            return_name=True,
        )
    )
    events = events.reset_index()
    events.columns = ['Date', 'Event']

    if additional_events is not None:
        assert additional_events.shape[1] == 2, 'additional events must be a df with 2 columns'
        additional_events.columns = ['Date', 'Event']
        additional_events['Date'] = pd.to_datetime(additional_events['Date'])
        events = events.append(additional_events)

    events['Date'] = [
        dt.datetime.strftime(pd.to_datetime(date), '%Y-%m-%d') for date in events['Date']
    ]
    return events.drop_duplicates().sort_values(by='Date').reset_index(drop=True)


def get_ts_calendar_from_project(project_id):
    return dr.CalendarFile.list(project_id=project_id)[0]


def create_and_upload_ts_calendar(
    df, ts_settings, filename='events_cal.csv', calendar_name='Calendar', calendar=None, multiseries_id= None):
    """
    df: pandas df
    ts_settings: dict
        Parameters for time series project
    calendar: pandas df (optional)
        If calendar is None a new calendar will be created
    multiseries_id: str
        Column name in calendar file that aligns with multiseries id
    Returns:
    --------
    DataRobot calendar object
    """
    if calendar is None:
        calendar = create_ts_calendar(df, ts_settings)
        calendar.to_csv(filename, index=False)
    else:
        calendar.to_csv(filename, index=False)

    print('Calendar file has been created')
    cal = dr.CalendarFile.create(file_path=filename, calendar_name=calendar_name, multiseries_id_columns=multiseries_id)
    print(f'Calendar file {cal.id} has been uploaded')

    return cal



def plot_ts_calendar(df, ts_settings, calendar=None):
    """
    Add calendar dates to plot of average target values
    
    df: pandas df
    ts_settings: dict
        Parameters of datetime DR project
    calendar: DataRobot calendar object
        if None, automatically creates calendar. Premade calendar can be shown instead
        
    Returns:
    --------
    Plotly lineplot with added calendar dates as scatter plot
    """

    date_col = ts_settings['date_col']
    target = ts_settings['target']
    df = df.copy()

    if calendar is None:
        calendar = create_ts_calendar(df=df, ts_settings=ts_settings)

    project_time_unit, project_time_step = get_timestep(df, ts_settings)

    # Lots of probably unnecessary code just to get a timedelta
    validation_durations = {'minute': 0, 'hour': 0, 'day': 0, 'month': 0}
    holdout_durations = {'minute': 0, 'hour': 0, 'day': 0, 'month': 0}

    if project_time_unit == 'minute':
        validation_durations['minute'] = ts_settings['validation_duration']
        holdout_durations['minute'] = ts_settings['holdout_duration']

    elif project_time_unit == 'hour':
        validation_durations['hour'] = ts_settings['validation_duration']
        holdout_durations['hour'] = ts_settings['holdout_duration']

    elif project_time_unit == 'day':
        validation_durations['day'] = ts_settings['validation_duration']
        holdout_durations['day'] = ts_settings['holdout_duration']

    elif project_time_unit == 'week':
        validation_durations['day'] = ts_settings['validation_duration'] * 7
        holdout_durations['day'] = ts_settings['holdout_duration'] * 7

    elif project_time_unit == 'month':
        validation_durations['day'] = ts_settings['validation_duration'] * 30
        holdout_durations['day'] = ts_settings['holdout_duration'] * 30

    else:
        raise ValueError(f'{project_time_step} {project_time_unit} is not a supported timestep')

    # Calculate a gap so we can drop observations from the holdout and validation folds
    gap = dt.timedelta(
        minutes=holdout_durations['minute'],
        hours=holdout_durations['hour'],
        days=holdout_durations['day'],
    ) + dt.timedelta(
        minutes=validation_durations['minute'],
        hours=validation_durations['hour'],
        days=validation_durations['day'],
    )

    # Subset to the training period
    max_date = ts_settings['max_date'] - gap
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.loc[df[date_col] <= max_date, :].copy()

    calendar['Date'] = pd.to_datetime(calendar['Date'])
    calendar.columns = ['cal_date', 'Event']

    # Calculate average target value per timestep
    df = df.groupby(date_col)[target].mean().reset_index()
    df[date_col] = pd.to_datetime(df[date_col])

    # Merge daily calendar events on original df
    calendar['_tmp_key'] = 1
    df['_tmp_key'] = 1
    merged = pd.merge(df, calendar, on='_tmp_key')

    # If the project time unit is weekly then merge daily calendar events to the nearest week
    if project_time_unit == 'week':
        epsilon = dt.timedelta(days=7)
    else:
        epsilon = dt.timedelta(days=0)

    merged = merged[
        (merged['cal_date'] >= merged[date_col])
        & (merged['cal_date'] <= merged[date_col] + epsilon)
    ].reset_index(drop=True)

    min_date = df[date_col].min()
    merged = merged[(merged['cal_date'] >= min_date) & (merged['cal_date'] <= max_date)]

    fig = px.line(df, x=date_col, y=target)

    fig.add_scatter(
        x=merged[date_col],
        y=merged[target],
        mode="markers+text",
        text=merged['Event'],
        showlegend=False,
    )
    fig.update_xaxes(title='Date')
    fig.update_traces(textposition='top center')
    # fig.update_layout(xaxis_rangeslider_visible = True)

    fig.show()
