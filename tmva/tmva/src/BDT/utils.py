import json
import os
import numpy as np

import xgboost as xgb
from xgboost import XGBClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import plot_tree
import matplotlib.pyplot as plt
import xgboost as xgb
import timeit


##### --------------------------------------------------------------------------
# BENCHMARK functions
##### --------------------------------------------------------------------------
def get_benchs_data(bench_fname):
    with open(bench_fname, "r") as file:
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
    num_samples,  # = 100
    num_features,  # =5,
    num_trees,  # =3,
    max_depth,  # =8,
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


##### --------------------------------------------------------------------------
# JSON handling functions
##### --------------------------------------------------------------------------
def read_json_xgb_model(filename):
    with open(filename, "r") as file:
        data = file.read().replace("\n", "")
    return json.loads(data)


def _test_if_valid_filename(models_filename):
    try:
        with open(models_filename, "r") as file:
            data = file.read().replace("\n", "")
            tmp_models = json.loads(data)
    except:
        tmp_models = dict()
        print(f"<{models_filename}> is in an invalid format")
    if not isinstance(tmp_models, dict):
        raise ValueError(f"File <{models_filename}> stores data in the wrong format")


def _is_valid_file(models_filename):
    is_valid = 1
    try:
        tmp_models = json.loads(models_filename)
    except:
        tmp_models = dict()
        is_valid = 0
    if not isinstance(tmp_models, dict):
        is_valid = 0
    if not os.path.isfile(models_filename):
        is_valid = 0
    return is_valid


def create_dict_from_xgboost_json(
    xgb_json_filename, model_key, objective_function="logistic", num_classes=1
):
    storage_dict = dict()
    model_tmp = read_json_xgb_model("regressed_model.json")
    storage_dict[model_key] = dict(
        model=model_tmp, objective_function="logistic", num_classes=num_classes
    )
    return storage_dict


def read_models_file_to_dict(models_filename):
    with open("test.json", "r") as file:
        data = file.read().replace("\n", "")
        aaa = json.loads(data)
    return aaa


def rewrite_xgb_json(
    xgb_json_filename,
    models_filename,
    model_key,
    objective_function="logistic",
    num_classes=1,
    overwrite=False,
):
    """
    :param xgb_json_filename: [string] path to json file with the the json dump of xgboost
    :param models_filename: [string] path to json file that can contains several Root forest models. /!\ Warning: the original file will be overwritten /!\
    :param model_key
    """
    if os.path.isfile(models_filename) and not overwrite:
        old_dict = read_models_file_to_dict(models_filename)
    else:
        old_dict = {}
    new_dict = create_dict_from_xgboost_json(
        xgb_json_filename, model_key, objective_function, num_classes
    )

    old_dict.update(new_dict)
    with open(models_filename, "w") as file:
        json.dump(old_dict, file, indent=3)
