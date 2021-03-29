import pandas as pd
import numpy as np
import os

from rainforest_audio.utilities.utilities import data_path


def load_dataset(filename):
    """Aggregates submission given filename"""
    temp = 0
    for i in [0, 1, 2, 3, 4]:
        sub = pd.read_csv(os.path.join(
            data_path, filename, f"submission{i}.csv"))
        temp += sub.iloc[:, 1:]
        temp /= 5
    return temp


if __name__ == "__main__":
    pass
