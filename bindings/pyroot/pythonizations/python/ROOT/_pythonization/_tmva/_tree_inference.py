# Author: Stefan Wunsch CERN  09/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from .. import pythonization
import cppyy

import json


def get_basescore(model):
    """Get base score from an XGBoost sklearn estimator.

    Copy-pasted from XGBoost unit test code.

    See also:
      * https://github.com/dmlc/xgboost/blob/a99bb38bd2762e35e6a1673a0c11e09eddd8e723/python-package/xgboost/testing/updater.py#L13
      * https://github.com/dmlc/xgboost/issues/9347
      * https://discuss.xgboost.ai/t/how-to-get-base-score-from-trained-booster/3192
    """
    base_score = float(json.loads(model.get_booster().save_config())["learner"]["learner_model_param"]["base_score"])
    return base_score


def SaveXGBoost(xgb_model, key_name, output_path, num_inputs):
    """
    Saves the XGBoost model to a ROOT file as a TMVA::Experimental::RBDT object.

    Args:
        xgb_model: The trained XGBoost model.
        key_name (str): The name to use for storing the RBDT in the output file.
        output_path (str): The path to save the output file.
        num_inputs (int): The number of input features used in the model.

    Raises:
        Exception: If the XGBoost model has an unsupported objective.
    """
    # Extract objective
    objective_map = {
        "multi:softprob": "softmax",  # Naming the objective softmax is more common today
        "binary:logistic": "logistic",
        "reg:linear": "identity",
        "reg:squarederror": "identity",
    }
    model_objective = xgb_model.objective
    if not model_objective in objective_map:
        raise Exception(
            'XGBoost model has unsupported objective "{}". Supported objectives are {}.'.format(
                model_objective, objective_map.keys()
            )
        )
    objective = cppyy.gbl.std.string(objective_map[model_objective])

    # Determine number of outputs
    num_outputs = xgb_model.n_classes_ if "multi:" in model_objective else 1

    # Dump XGB model as json file
    xgb_model.get_booster().dump_model(output_path, dump_format="json")

    with open(output_path, "r") as json_file:
        forest = json.load(json_file)

    # Dump XGB model as txt file
    xgb_model.get_booster().dump_model(output_path)

    features = cppyy.gbl.std.vector["std::string"]([f"f{i}" for i in range(num_inputs)])
    bs = get_basescore(xgb_model)
    logistic = objective == "logistic"
    bdt = cppyy.gbl.TMVA.Experimental.RBDT.LoadText(
        output_path,
        features,
        num_outputs,
        logistic,
        cppyy.gbl.std.log(bs / (1.0 - bs)) if logistic else bs,
    )

    with cppyy.gbl.TFile.Open(output_path, "RECREATE") as tFile:
        tFile.WriteObject(bdt, key_name)
