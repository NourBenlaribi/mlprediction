import argparse

parser = argparse.ArgumentParser(description='Machine learning applied to hard drives failures')
parser.add_argument('--fill_by', help='how to fill the missing args')
parser.add_argument('--iterations', help='how many iterations', default=180, type=int)
parser.add_argument('--window_size_from', help='minimum window size', default=1, type=int)
parser.add_argument('--window_size_to', help='maximum window size', default=20, type=int)
parser.add_argument('--use_smoothing', help='use exponential smoothing')
parser.add_argument('--use_small_data', help='use small dataset')
parser.add_argument('--out', help='output folder', default='results')
args = parser.parse_args()
print(args)
fill_by = args.fill_by
use_smoothing = args.use_smoothing == 'True'
use_small_data = args.use_small_data == 'True'
window_size_from = args.window_size_from
window_size_to = args.window_size_to
iterations = args.iterations
out = args.out
