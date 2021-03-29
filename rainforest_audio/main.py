import pandas as pd
import numpy as np
import os

from rainforest_audio.utilities.utilities import data_path


def load_dataset(filename, num_sub=5):
    """Aggregates submission given filename"""
    temp = 0
    for i in np.arange(num_sub):
        sub = pd.read_csv(os.path.join(
            data_path, filename, "submission{i}.csv".format(i=i)))
        temp += sub.iloc[:, 1:] / 5
        del sub
    return temp


if __name__ == "__main__":
    all_dense = 0
    for filename in ["dense1", "dense2", "dense3", "dense4", "dense5", "dense6", "dense7"]:
        all_dense += load_dataset(filename, num_sub=5)
    all_dense /= 7

    kernel = pd.read_csv(os.path.join(data_path, "kernel", "kernel_088.csv"))
    final = (all_dense * 0.95) + (0.05 * kernel.iloc[:, 1:])
    final['recording_id'] = kernel['recording_id']
    final.to_csv(os.path.join(
        data_path, "final_ensemble_submission.csv"), index=False)
