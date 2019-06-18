from arguments import fill_by, use_smoothing, use_small_data, iterations, window_size_from, window_size_to, out, use_all_data
from datetime import datetime
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline

from data_loader import load_data_frame_2014, load_small_data_frame
from feature_selection import COLUMNS
from pre_processing import flag_failures, clean_data
from sampling import split_to_folds
from matplotlib import pyplot as plt

plt.style.use('seaborn-whitegrid')
import pandas as pd
from fill_na.fill_by_drop import drop_na
from fill_na.fill_by_frequency import fill_by_frequency
from fill_na.fill_by_mean import fill_by_mean
from fill_na.fill_by_zero import fill_by_zero
from smoothing_smart_param import smooth

print("##### loading data #####")
if use_small_data:
    df = load_small_data_frame()
else:
    df = load_data_frame_2014()

print("##### sorting data #####")
df.sort_values(by=['serial_number', 'date'], ascending=[True, False], inplace=True)

print(f"##### fill NA by ({fill_by}) #####")
if fill_by == 'drop':
    df = drop_na(df)
if fill_by == 'zero':
    df = fill_by_zero(df)
elif fill_by == 'mean':
    df = fill_by_mean(df)
elif fill_by == 'frequency':
    df = fill_by_frequency(df)
else:
    raise Exception('error: fill method not known')
df.reset_index(drop=True, inplace=True)

if not use_all_data:
    print("##### pre-processing #####")
    df = clean_data(df)

if use_smoothing:
    print("##### applying exponential smoothing #####")
    smooth(df)


def save_results(d, iteration, window_size):
    outdir = f'./{out}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results_filename = f'{outdir}/iteration_{iteration}_window_size_{window_size}'
    d.to_csv(f'{results_filename}.csv', index=False)


results = pd.DataFrame(columns=['window_size', 'iteration', 'fold', 'accuracy', 'precision', 'recall'])
for window_size in range(window_size_from, window_size_to + 1):
    print(f"##### window size {window_size} #####")
    split_to_folds(df)
    copy = df.copy()
    flag_failures(copy, time_window_size=window_size)
    for i in range(0, iterations):
        print(f"##### iteration {i} #####")
        classifier = RandomForestClassifier(n_estimators=50, max_depth=8)
        for fold in range(3):
            x_test = copy[copy['fold'] == fold]
            x_train = copy[copy['fold'] != fold]

            y_train = x_train['failure'].astype('int')
            y_test = x_test['failure'].astype('int')
            x_test = x_test[COLUMNS]
            x_train = x_train[COLUMNS]

            pipeline = make_pipeline(classifier)
            model = pipeline.fit(x_train, y_train)
            prediction = model.predict(x_test)

            accuracy = pipeline.score(x_test, y_test)
            precision = precision_score(y_test, prediction)
            recall = recall_score(y_test, prediction)
            results = results.append({'window_size': window_size,
                                      'iteration': i,
                                      'fold': fold,
                                      'accuracy': accuracy,
                                      'precision': precision,
                                      'recall': recall}, ignore_index=True)
        save_results(results, i, window_size)
