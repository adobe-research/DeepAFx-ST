import os
import sys
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("input", help="Path to results .json file.", type=str)
args = parser.parse_args()

with open(args.input) as fp:
    results = json.load(fp)

for method, metrics in results.items():
    sys.stdout.write(f"{method:20s} ")
    for metric_name, values in metrics.items():
        sys.stdout.write(f"{np.mean(values):0.3f} & ")
    print()
    sys.stdout.flush()
