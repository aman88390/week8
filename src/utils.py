import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path, test_split=0.3, random_state=42):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    return train_test_split(X, y, test_size=test_split, random_state=random_state)
