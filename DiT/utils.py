import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd

def aggregate_results(dir):
    glob_name = os.path.join(dir, "**/**/*.log")
    paths = glob.glob(glob_name)

    # extract results from log files
    pattern = r'(.+):\s*([0-9]+(?:\.[0-9]+)?)'
    results = []
    for path in paths:
        with open(path, "r") as f:
            logs = f.readlines()
        result = re.findall(pattern, "".join(logs))
        
        result = dict(result)
        result = dict((k, float(v)) for (k,v) in result.items())

        parts = Path(path).parts
        result["scheduler"] = parts[-3]
        result["order"] = float(parts[-2].replace("_", "."))
        result["step"] = parts[-1].split(".")[0]
        results.append(result)

    # export as csv file
    df = pd.DataFrame.from_dict(results).sort_values(by=["step"]).reset_index(drop=True)
    df = df.sort_values(by=["order", "step"]).reset_index(drop=True)
    
    save_path = os.path.join(dir, "results.csv")
    df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    args = parser.parse_args()

    aggregate_results(args.dir)