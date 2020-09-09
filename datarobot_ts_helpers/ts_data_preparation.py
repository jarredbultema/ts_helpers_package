import pandas as pd
import numpy as np
import psycopg2 as pg

def execute_query(statement, conn_string):
    '''
    Executes a Query on a SQL database
    
    statement: str
        SQL query
    conn_string: str
        Connection string for the database of interest
        
    Returns:
    --------
    output of SQL query
    '''
    
    # Construct connection string
#     conn_string = f"host={host} user={user} dbname={dbname} password={password} sslmode={sslmode}"
    conn = pg.connect(conn_string) 
    print("Connection established")

    try: 
        cursor = conn.cursor()
        cursor.execute(statement)
        return cursor.fetchall()
    except: 
        return "No results to fetch"
    
    
def create_df_from_query(table_name, cursor, index_col= None, select_cols= '*'):
    '''
    Create a Pandas DataFrame from a SQL query when you want select columns from an existing table
    
    table_name: str
        Name of table in database
    cursor: psycopg2.extensions.cursor
        Psycopg2 cursor object
    index_col: str
        Column name for index column
    select_cols: str
        Columns to be selected from table, must be in SQL-sytnax
    
    Returns:
    --------
    Pandas df
    '''
    
    statement = f'''
    SELECT {select_cols} 
    from {table_name};'''
    cursor.execute(statement)
    out1 = cursor.fetchall()

    statement2 = f"""
    SELECT COLUMN_NAME
    FROM information_schema.COLUMNS
    WHERE TABLE_NAME = '{table_name}';"""
    cursor.execute(statement2) 
    out2 = cursor.fetchall()
    names = [x[0] for x in out2]

    # create the dataframe
    return pd.DataFrame.from_records(out1, index= index_col, columns= names) # 

def aggregate_df(df, aggregators= None, string_columns= None, numeric_columns= None, ignore_columns= None):
    '''
    Aggregates individual data by designated single or multi-level aggregation columns and performs aggregation for different column types.
    -----
    inputs:
    df: Pandas df
        input df with non-aggregated timeseries data
    aggregators: list
        list of columnms names to be used for aggregation
    string_columns: list
        list of columnms names that should be treated as strings
    numeric_columns: list 
        list of columnms names that should be treated as numeric (int/float) objects
    ignore_columns: list
        list of columns that should not have aggregation statistics calculated
        
    Returns
    -------
    Pandas df
    '''
    
    # check types
    for item in [aggregators, string_columns, numeric_columns, ignore_columns]:
        if len(item) > 0:
            assert isinstance(item, list), 'Parameters must be a list'
#                 return None
    
    # create empty output dataframe
    out_df = pd.DataFrame()
    
    # create string aggregate columns
    print(f'Processing {len(string_columns)} string columns:')
    
    for si, col in enumerate(string_columns, 1):
        print(f'*** Processing {col} {si}/{len(string_columns)} ***')
        min_name = str(col) + "_min"
        max_name = str(col) + "_max"
        unique_name = str(col) + "_unique"
        # column needed to set the proper index
        out_df['dummy'] = df.groupby(aggregators)[col].nth(0) 
        try:
            # this causes an error with np.nan values
            out_df[min_name] = df.groupby(aggregators).agg({col: lambda x: pd.Series.value_counts(x).idxmin()}) 
        except: 
            # workaround: take the first row value
            out_df[min_name] = df.groupby(aggregators)[col].nth(0) 
        try: 
            # this causes an error with np.nan values
            out_df[max_name] = df.groupby(aggregators).agg({col: lambda x: pd.Series.value_counts(x).idxmax()}) 
        except: 
            # workaround: take the last row value
            out_df[max_name] = df.groupby(aggregators)[col].nth(-1) 
            
        out_df[unique_name] = df.groupby(aggregators).agg({col: pd.Series.nunique})
        
        # drop the 'dummy' column used to set the index
        out_df.drop('dummy', axis=1, inplace= True) 
        
    # create numeric aggregated columns    
    print(f'\nProcessing {len(numeric_columns)} numeric columns:')
    
    for ni, n_col in enumerate(numeric_columns, 1):
        print(f'*** Processing {n_col} {ni}/{len(numeric_columns)} ***')
        
        # create the computed columns
        out_df[str(n_col) + "_min"] = df.groupby(aggregators).agg({n_col: min})
        out_df[str(n_col) + "_mean"] = df.groupby(aggregators).agg({n_col: np.mean})
        out_df[str(n_col) + "_max"] = df.groupby(aggregators).agg({n_col: max})
        out_df[str(n_col) + "_stdev"] = df.groupby(aggregators).agg({n_col: pd.Series.std})
        out_df[str(n_col) + "_unique"] = df.groupby(aggregators).agg({n_col: pd.Series.nunique})
        out_df[str(n_col) + "_sum"] = df.groupby(aggregators).agg({n_col: sum})
        
    # copy simple or excluded columns
    print(f'\nProcessing {len(ignore_columns)} simple columns:')
    
    for ii, i_col in enumerate(ignore_columns, 1):
        print(f'*** Processing {i_col} {ii}/{len(ignore_columns)} ***')
        try:
            # when possible, the mode is used to select the most common value
            out_df[i_col] = df.groupby(aggregators).agg({i_col: pd.Series.mode})
        except:
            # when note possible, the mean is used
            out_df[i_col] = df.groupby(aggregators).agg({i_col: pd.Series.mean})
        
    # create a column to capture total daily transactions    
    out_df['aggregated_total_count'] = df.groupby(aggregators).agg({string_columns[0]: 'size'})
    
    return out_df.reset_index()