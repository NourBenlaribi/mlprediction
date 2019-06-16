import numpy as np


def split_to_folds(df, fold_count=3):
    df['fold'] = 0
    hdd_fold = 0
    current_hdd = None
    for tpl in df.itertuples():
        if current_hdd != tpl.serial_number:
            current_hdd = tpl.serial_number

            hdd_fold = np.random.randint(low=0, high=fold_count)
        df.at[tpl.Index, 'fold'] = hdd_fold
