from feature_selection import COLUMNS
import pandas as pd
import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing


def smooth(df, window_size):
    current_hdd = None
    new_df = pd.DataFrame(columns=COLUMNS)
    for tpl in df.itertuples():
        serial_number = tpl.serial_number
        row = []
        if current_hdd != serial_number:
            current_hdd = serial_number
            for colomn in COLUMNS:
                if (tpl[0] == 0):
                        data = df[colomn].loc[tpl.Index:tpl.Index + 6]
                if (tpl[0] == 1):
                        data = df[colomn].loc[tpl.Index:tpl.Index + window_size]
                data = np.array(data)
                paramdata = pd.Series(data)
                fit1 = SimpleExpSmoothing(paramdata).fit(smoothing_level=0.2, optimized=False)
                fcast1 = fit1.forecast(1).rename(r'$\alpha=0.1$')
                elem = fit1.fittedvalues.iloc[-1]
                row.append(elem)
            new_df = new_df.append({COLUMNS[0]: row[0], COLUMNS[1]: row[1], COLUMNS[2]: row[2], COLUMNS[3]: row[3], COLUMNS[4]: row[4],COLUMNS[5]: row[5], COLUMNS[6]: row[6], COLUMNS[7]: row[7] }, ignore_index=True)
    print(new_df)
