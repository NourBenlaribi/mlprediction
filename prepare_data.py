import argparse

parser = argparse.ArgumentParser(description='Machine learning applied to hard drives failures')
parser.add_argument('--fill_by', help='how to fill the missing args', required=True)
parser.add_argument('--use_all_data', help='use all data')
parser.add_argument('--window_size_from', help='minimum window size', default=1, type=int)
parser.add_argument('--window_size_to', help='maximum window size', default=20, type=int)
parser.add_argument('--use_smoothing', help='use exponential smoothing')
parser.add_argument('--use_small_data', help='use small dataset')
parser.add_argument('--out', help='output folder', required=True)

args = parser.parse_args()
print(args)
fill_by = args.fill_by
use_smoothing = args.use_smoothing == 'True'
use_small_data = args.use_small_data == 'True'
window_size_from = args.window_size_from
window_size_to = args.window_size_to
out = args.out
use_all_data = args.use_all_data == 'True'



import os

from data_loader import load_data_frame_2014, load_small_data_frame
from fill_na.fill_by_drop import drop_na
from fill_na.fill_by_frequency import fill_by_frequency
from fill_na.fill_by_mean import fill_by_mean
from fill_na.fill_by_zero import fill_by_zero
from pre_processing import flag_failures, clean_data
from smoothing_smart_param import smooth

print("##### loading data #####")
if use_small_data:
    print("##### loading small data #####")
    df = load_small_data_frame()
else:
    print("##### loading all data #####")
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


def save_data(d, window_size):
    outdir = f'./{out}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results_filename = f'{outdir}/window_size_{window_size}'
    d.to_csv(f'{results_filename}.csv', index=False)


def prepare_data():
    for window_size in range(window_size_from, window_size_to + 1):
        print(f"##### window size {window_size} #####")
        copy = df.copy()
        flag_failures(copy, time_window_size=window_size)
        save_data(copy, window_size)


prepare_data()
