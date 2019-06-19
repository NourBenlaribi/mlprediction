

def fill_by_mean(df):
    df["smart_5_raw"].fillna(0, inplace = True)
    df["smart_12_raw"].fillna(1.576954e+02 , inplace = True)
    df["smart_187_raw"].fillna(1, inplace = True)
    df["smart_188_raw"].fillna(2.32170e+10 , inplace = True)
    df["smart_189_raw"].fillna(48, inplace = True)
    df["smart_190_raw"].fillna(24, inplace = True)
    df["smart_198_raw"].fillna(3, inplace = True)
    df["smart_199_raw"].fillna(48, inplace = True)
    df["smart_200_raw"].fillna(0, inplace = True)
    return df
