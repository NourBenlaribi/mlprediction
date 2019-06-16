def clean_data(df, to_keep_healthy=7, to_keep_failure=20):
    print('the begining of clean_data ')
    df['keep'] = False
    current_hdd = None
    to_keep = 0
    for tpl in df.itertuples(): # range(row_size):
        serial_number = tpl.serial_number
        if current_hdd != serial_number:
            current_hdd = serial_number
            failure = tpl.failure
            if failure == 0 and to_keep == 0:
                to_keep = to_keep_healthy + 1
            elif failure == 1 and to_keep == 0:
                to_keep = to_keep_failure + 1
        df.at[tpl.Index, 'keep'] = to_keep != 0
        if to_keep != 0:
            to_keep -= 1
    print('reset index ')
    df = df[df['keep'] == True]
    df = df.drop('keep', axis=1)
    df.reset_index(drop=True, inplace=True)
    return df


def flag_failures(df, time_window_size):
    i = 0
    failure = False
    current_hdd = None
    for tpl in df.itertuples():
        if current_hdd != tpl.serial_number:
            current_hdd = tpl.serial_number
            i = 0
            failure = False
        if tpl.failure == 1:
            i = time_window_size
            failure = True
        elif (failure is True) and (i > 0):
            df.at[tpl.Index, 'failure'] = 1
            i -= 1
