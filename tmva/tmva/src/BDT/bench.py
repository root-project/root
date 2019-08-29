import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
import matplotlib.pyplot as plt
import xgboost as xgb
import timeit
import os
import subprocess

import json  # for testing jsonness

from utils import *

DATA_FOLDER = "./data/"


def execute_bench(TMP_FOLDER, test_list, repetitions, kwargs):
    # TMP_FOLDER = "./tmp/timeVSdepth/"
    if not os.path.exists(TMP_FOLDER):
        os.makedirs(TMP_FOLDER)
    onlyfiles = sorted(
        [
            f
            for f in os.listdir(TMP_FOLDER)
            if os.path.isfile(os.path.join(TMP_FOLDER, f))
        ]
    )
    for the_file in onlyfiles:
        if the_file[-4:] == ".npy":
            os.remove(TMP_FOLDER + the_file)

    # test_list = np.arange(5, 8, 3)
    np.save(TMP_FOLDER + "0_abscisse.npy", np.array(test_list))
    for i, value_test in enumerate(test_list):
        print(f"***** bench value {value_test} *****")
        mins_list = []
        for j in range(repetitions):
            create_model_gaussian(max_depth=value_test, **kwargs)
            subprocess.call("./bench.sh".split(), shell=True)
            fname, mins, means, stddevs = get_benchs_data("./data/a.txt")
            mins_list.append(mins)
        np.save(
            TMP_FOLDER + "min_" + "{0:03}".format(i) + "_",
            np.min(np.stack(mins_list), axis=0),
        )
        np.save(
            TMP_FOLDER + "std_" + "{0:03}".format(i) + "_",
            np.std(np.stack(mins_list), axis=0),
        )


timeVSdepth = dict(
    num_samples=100_000,
    num_features=10,
    num_trees=250,
    data_folder=DATA_FOLDER,
    save_models=True,
)
timeVSfeats = dict(
    num_samples=100_000,
    max_depth=3,
    num_trees=100,
    data_folder=DATA_FOLDER,
    save_models=True,
)
timeVSevents = dict(
    num_features=10,
    max_depth=3,
    num_trees=250,
    data_folder=DATA_FOLDER,
    save_models=True,
)
timeVStrees = dict(
    num_samples=100_000,
    num_features=10,
    max_depth=3,
    data_folder=DATA_FOLDER,
    save_models=True,
)

if __name__ == "__main__":
    # timeVSdepth
    execute_bench(
        "./tmp/timeVSdepth/", [2, 3, 4, 5, 6, 7], 3, timeVSdepth  # repetitions
    )

    execute_bench(
        "./tmp/timeVSfeats/", [5, 15, 25, 35, 45, 55], 4, timeVSfeats  # repetitions
    )
    execute_bench(
        "./tmp/timeVStrees/", [100, 300, 500, 700, 900], 2, timeVStrees  # repetitions
    )
    execute_bench(
        "./tmp/timeVSfewEvents/",
        [2, 4, 6, 8, 10, 12, 14, 16],
        10,  # repetitions
        timeVSevents,
    )
    execute_bench(
        "./tmp/timeVSmiddleEvents/",
        [20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        5,  # repetitions
        timeVSevents,
    )
    execute_bench(
        "./tmp/timeVSevents/",
        [1000, 10000, 100_000, 300_000, 500_000],
        2,  # repetitions
        timeVSevents,
    )
    # """


# end
