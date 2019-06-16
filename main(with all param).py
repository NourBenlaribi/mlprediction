from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import make_pipeline

from data_loader import load_data_frame_2014, load_small_data_frame
from feature_selection import COLUMNS
from pre_processing import flag_failures, clean_data
from sampling import split_to_folds
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
import pandas as pd

print("##### loading data #####")
df = load_data_frame_2014()

print("##### sorting data #####")
df.sort_values(by=['serial_number', 'date'], ascending=[True, False], inplace=True)

print("##### cleaning data #####")
df = df.fillna(0)
df.reset_index(drop=True, inplace=True)

print("##### pre-processing #####")
df = clean_data(df)
df.to_csv('./data/data_2014_replacing_by_zero.csv')

df = pd.read_csv('./data/data_2014_replacing_by_zero.csv')

global_accuracy = []
global_precision = []
global_recall = []

for window_size in range(1, 21):
    print("##### iteration #####", window_size)
    print("##### create folds #####")
    split_to_folds(df)
    copy = df.copy()
    flag_failures(copy, time_window_size=window_size)
    classifier = RandomForestClassifier(n_estimators=50, max_depth=8)
    precision_i = []
    accuracy_i = []
    recall_i = []
    for i in range(0, 180):
        precision = []
        accuracy = []
        recall = []

        for fold in range(3):
            x_test = copy[copy['fold'] == fold]
            x_train = copy[copy['fold'] != fold]

            y_train = x_train['failure'].astype('int')
            y_test = x_test['failure'].astype('int')
            x_test = x_test[COLUMNS]
            x_train = x_train[COLUMNS]

            print(f"##### train model for {fold} #####")
            pipeline = make_pipeline(classifier)
            model = pipeline.fit(x_train, y_train)
            prediction = model.predict(x_test)

            accuracy.append(pipeline.score(x_test, y_test))
            precision.append(precision_score(y_test, prediction))
            recall.append(recall_score(y_test, prediction))
        print(f'accuracy= {accuracy}')
        print(f'precision= {precision}')
        print(f'recall= {recall}')
        print(f'mean accuracy= {np.mean(accuracy)}')
        print(f'mean precision= {np.mean(precision)}')
        print(f'mean recall= {np.mean(recall)}')
        accuracy_i.append(np.mean(accuracy))
        precision_i.append(np.mean(precision))
        recall_i.append(np.mean(recall))

    global_accuracy.append(np.mean(accuracy_i))
    global_precision.append(np.mean(precision_i))
    global_recall.append(np.mean(recall_i))

x = [i for i in range(1, 21)]
print(f'x={x}')
print(f'global_accuracy={global_accuracy}')
print(f'global_precision={global_precision}')
print(f'global_recall={global_recall}')



plt.title("Precision with replacing nan value by zero")
plt.xlabel('Time window lenght(days)')
plt.ylabel('Precision')
plt.xticks([i for i in range(1,21)])
plt.yticks([i*0.1 for i in range(1,10)])
plt.plot(x, global_precision)
plt.show()

plt.title("Recall with replacing nan value by zero")
plt.xlabel('Time window lenght(days)')
plt.ylabel('Recall')
plt.xticks([i for i in range(1,21)])
plt.yticks([i*0.1 for i in range(1,10)])
plt.plot(x, global_recall)
plt.show()

