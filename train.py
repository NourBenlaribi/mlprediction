# import argparse
#
# parser = argparse.ArgumentParser(description='Machine learning applied to hard drives failures')
# parser.add_argument('--input', help='data file', required=True)
# parser.add_argument('--out', help='output folder', required=True)
# parser.add_argument('--window', help='window', required=True)
# parser.add_argument('--iteration', help='iteration', required=True)
#
# args = parser.parse_args()
# print(args)
# window = args.window
# iteration = args.iteration
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline

from feature_selection import COLUMNS
from sampling import split_to_folds

import pandas as pd


def save_results(d, out, iteration, window):
    outdir = f'./{out}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results_filename = f'{outdir}/iteration_{iteration}_window_{window}'
    d.to_csv(f'{results_filename}.csv', index=False)


def train(input, out, window, iteration):
    print(f"##### iteration {iteration}. window size {window} #####")
    print("##### loading data #####")
    df = pd.read_csv(input)
    results = pd.DataFrame(columns=['window_size', 'iteration', 'fold', 'accuracy', 'precision', 'recall'])
    classifier = RandomForestClassifier(n_estimators=50, max_depth=8)
    split_to_folds(df)
    for fold in range(3):
        x_test = df[df['fold'] == fold]
        x_train = df[df['fold'] != fold]
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
        results = results.append({'window_size': window,
                                  'iteration': iteration,
                                  'fold': fold,
                                  'accuracy': accuracy,
                                  'precision': precision,
                                  'recall': recall}, ignore_index=True)

    save_results(results, out, iteration, window)
