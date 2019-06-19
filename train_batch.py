import argparse
# python train_batch.py --parallel=4 --input=data/mean_all_data --out=results/mean_all_data --out_final=C:\Users\noure\OneDrive\results\mean_all_data
parser = argparse.ArgumentParser(description='Machine learning applied to hard drives failures')
parser.add_argument('--parallel', help='process count', type=int, required=True)
parser.add_argument('--input', help='input data folder', required=True)
parser.add_argument('--out', help='ouput folder', required=True)
parser.add_argument('--out_final', help='ouput aggregated results folder', required=True)
args = parser.parse_args()

from multiprocessing import Process
from train import train
import pandas as pd
import queue
import os


def train_for_window(iteration):
    for window in range(1, 21):
        train(f'{args.input}/window_size_{window}.csv', f'{args.out}', window, iteration)


def save_results(d, out, iteration):
    outdir = f'{out}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    results_filename = f'{outdir}/iteration_{iteration}'
    d.to_csv(f'{results_filename}.csv', index=False)


def run():
    iterations = queue.Queue()
    [iterations.put(i) for i in range(180)]
    results = pd.DataFrame(columns=['window_size', 'iteration', 'fold', 'accuracy', 'precision', 'recall'])
    while not iterations.empty():
        processes = []
        processed_iterations = []
        for i in range(args.parallel):
            iteration = iterations.get()
            processed_iterations.append(iteration)
            process = Process(target=train_for_window, args=(iteration,))
            process.start()
            processes.append(processes)
            if iterations.empty():
                break

        for p in processes:
            p.join()

        for it in processed_iterations:
            for window in range(1, 21):
                df = pd.read_csv(f'{args.out}/iteration_{it}_window_{window}')
                results = results.append(df)
        save_results(results, args.out_final, processed_iterations[-1])


if __name__ == '__main__':
    run()
