import pandas as pd
from calendar import monthrange

from feature_selection import COLUMNS


def read_day(day, month, year, columns=COLUMNS):
    return pd.read_csv(f'./{year}/{year}-{str(month).zfill(2)}-{str(day).zfill(2)}.csv', usecols=['failure',
                                                                                                  'date',
                                                                                                  'serial_number',
                                                                                                  *columns])

def read_month(month, year, columns=COLUMNS):
    df = read_day(1, month, year, columns)
    rg = monthrange(year, month)
    for i in range(2, rg[1] + 1):
        df = pd.concat([df, read_day(i, month, year, columns)])
    return df


def read_year(year, columns=COLUMNS):
    df = read_month(4, year, columns)
    for i in range(5, 13):
        df = pd.concat([df, read_month(i, year, columns)])
    return df


def load_data_frame_2014(columns=COLUMNS):
    return read_year(2014, columns)


def load_small_data_frame(columns=COLUMNS):
    return read_month(12, 2014, columns)

