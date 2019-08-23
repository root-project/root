import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
import matplotlib.pyplot as plt
import xgboost as xgb
import timeit

import subprocess

import json  # for testing jsonness

DATA_FOLDER = "./data/"


def get_benchs_data(bench_name="benchs/a.txt"):
    with open(bench_name, "r") as file:
        data = file.read().replace("\n", "")
    benchs = json.loads(data)
    fname = benchs["context"]["date"].replace(" ", "")
    mins = []
    means = []
    stddev = []
    for i, bench in enumerate(benchs["benchmarks"]):
        if bench["aggregate_name"] == "min":
            mins.append(bench["cpu_time"])
        if bench["aggregate_name"] == "stddev":
            stddev.append(bench["cpu_time"])
        if bench["aggregate_name"] == "mean":
            means.append(bench["cpu_time"])
    return fname, mins, means, stddev


def create_model_gaussian(
    num_samples=100,
    num_features=5,
    num_trees=3,
    max_depth=8,
    data_folder="./data/",
    save_models=True,
):
    mu, sigma = 0, 1  # mean and standard deviation

    training_samples = max(1000, num_samples)
    X = np.random.normal(mu, sigma, (training_samples, num_features))

    p = 0.5
    Y = np.random.choice(a=[0, 1], size=(training_samples), p=[p, 1 - p])

    # fit model no training data
    model = xgb.XGBClassifier(n_estimators=num_trees, max_depth=max_depth)
    model.fit(X, Y)

    X = X[:num_samples]
    Y = Y[:num_samples]

    y_pred = model.predict(X)
    y_scores = model.apply(X)

    # print(y_pred)
    predictions = [round(value) for value in y_pred]

    # evaluate predictions
    accuracy = accuracy_score(Y, predictions)
    print("For curiosity: Accuracy: %.2f%%" % (accuracy * 100.0))

    # saving files
    np.savetxt(data_folder + "events.csv", X, delimiter=",", fmt="%f")
    np.savetxt(data_folder + "python_predictions.csv", y_pred, delimiter=",", fmt="%d")
    np.savetxt(data_folder + "python_groundtruths.csv", Y, delimiter=",", fmt="%d")
    if save_models is True:
        model.get_booster().dump_model(data_folder + "model.json", dump_format="json")
        model.save_model(data_folder + "model.rabbit")
    print("Saved files")


def bench_1():
    test_list = [1, 10, 30, 40, 50, 60, 70, 100, 120, 130, 150, 160, 180, 200]
    offset = 0
    for i, num_samples in enumerate(test_list):
        create_model_gaussian(num_samples=num_samples, num_features=5, num_trees=100)
        subprocess.call("./bench.sh".split(), shell=True)
        fname, mins, means, stddevs = get_benchs_data("benchs/a.txt")
        # np.save("tmp/"+"{0:03}".format(i)+"_"+fname, mins)
        np.save("tmp/" + "{0:03}".format(i + offset) + "_", mins)


def bench_2():
    test_list = np.arange(50, 1500, 50)
    np.save("list.npy", test_list)
    offset = 0
    i = 1
    for i, num_trees in enumerate(test_list):
        mins_list = []
        for j in range(4):
            create_model_gaussian(
                num_samples=100_000,
                num_features=5,
                num_trees=num_trees,
                data_folder="./data/",
            )
            subprocess.call("./bench.sh".split(), shell=True)
            fname, mins, means, stddevs = get_benchs_data("benchs/a.txt")
            # np.save("tmp/"+"{0:03}".format(i)+"_"+fname, mins)
            mins_list.append(mins)
        np.save(
            "tmp/" + "{0:03}".format(i + offset) + "_",
            np.min(np.stack(mins_list), axis=0),
        )


if __name__ == "__main__":
    # bench_1()
    # bench_2()
    test_list = [5]
    # test_list = np.arange(5, 8, 3)
    np.save("tmp/0_abscisse.npy", np.array(test_list))
    offset = 0
    i = 1
    for i, value_test in enumerate(test_list):
        print(f"***** bench value {value_test} *****")
        mins_list = []
        create_model_gaussian(
            num_samples=10000,
            num_features=value_test,
            num_trees=3000,
            max_depth=3,
            data_folder=DATA_FOLDER,
            save_models=True,
        )
        subprocess.call("./bench.sh".split(), shell=True)
        fname, mins, means, stddevs = get_benchs_data("./data/a.txt")
        # np.save("tmp/"+"{0:03}".format(i)+"_"+fname, mins)
        mins_list.append(mins)
        np.save(
            "tmp/" + "{0:03}".format(i + offset) + "_",
            np.min(np.stack(mins_list), axis=0),
        )
        np.save(
            "tmp/std_" + "{0:03}".format(i + offset) + "_",
            np.std(np.stack(mins_list), axis=0),
        )


# end
