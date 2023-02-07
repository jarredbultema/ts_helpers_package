

### Take input df and comapre two columns
def compare_two_columns(df, col1, col2):
    df['compare'] = df[col1] == df[col2]
    return df