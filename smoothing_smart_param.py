from feature_selection import COLUMNS
import pandas as pd


def smooth(df, columns=COLUMNS, smoothing_level=0.2):
    current_hdd = None
    ft = pd.DataFrame(columns=columns, dtype=float)
    ft.append(list(map(lambda e: 0, columns)))
    at = pd.DataFrame(columns=columns, dtype=float)
    at.append(list(map(lambda e: 0, columns)))
    for tpl in df.itertuples():
        serial_number = tpl.serial_number
        if current_hdd != serial_number:
            current_hdd = serial_number
            for column in columns:
                ft.at[0, column] = df.at[tpl.Index, column]
                at.at[0, column] = df.at[tpl.Index, column]
        else:
            for column in columns:
                ft_1 = exp_smothing(ft.at[0, column], at.at[0, column], alpha=smoothing_level)
                ft.at[0, column] = ft_1
                at.at[0, column] = df.at[tpl.Index, column]
                df.at[tpl.Index, column] = ft_1


def exp_smothing(ft, at, alpha=0.2):
    return alpha * at + (1 - alpha) * ft
