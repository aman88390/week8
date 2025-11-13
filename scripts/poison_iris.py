import pandas as pd
import numpy as np
import argparse

def poison_data(df, rate):
    n = len(df)
    k = int(n * rate)

    print(f"Poisoning {k} samples ({rate*100}%)")

    idx = np.random.choice(n, k, replace=False)
    feature_cols = df.columns[:-1]
    label_col = df.columns[-1]

    mins = df[feature_cols].min()
    maxs = df[feature_cols].max()

    df_poisoned = df.copy()

    for i in idx:
        df_poisoned.loc[i, feature_cols] = np.random.uniform(mins, maxs)

    return df_poisoned

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rate", type=float, default=0.05)
    args = parser.parse_args()

    df = pd.read_csv("data/iris_clean.csv")
    poisoned = poison_data(df, args.rate)

    poisoned.to_csv("data/iris_poisoned.csv", index=False)
    print("Saved poisoned dataset → data/iris_poisoned.csv")
