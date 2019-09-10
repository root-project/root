# Author: Stefan Wunsch CERN  09/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from ROOT import pythonization
import cppyy


def SaveXGBoost(self, xgb_model, key_name, output_path, tmp_path = "/tmp", threshold_dtype="float"):
    # Extract parameters from the model object
    max_depth = xgb_model.max_depth
    num_features = xgb_model._features_count
    num_classes = xgb_model.n_classes_
    objective_map = {
            "binary:logistic": "logistic",
            }
    model_objective = xgb_model.objective
    if not model_objective in objective_map:
        raise Exception('XGBoost model has unsupported objective "{}". Supported objectives are {}.'.format(
            model_objective, objective_map.keys()))
    objective = cppyy.gbl.std.string(objective_map[model_objective])

    # Dump XGB model to the tmp folder as json file
    import os
    import uuid
    tmp_path = os.path.join(tmp_path, str(uuid.uuid4()) + ".json")
    xgb_model.get_booster().dump_model(tmp_path, dump_format="json")

    # Extract parameters from json and write to arrays
    import json
    forest = json.load(open(tmp_path, "r"))
    #print(str(forest).replace("u'", "'").replace("'", '"'))
    num_trees = len(forest)
    len_features = 2**max_depth - 1
    features = cppyy.gbl.std.vector["int"](len_features * num_trees, -1)
    len_thresholds = 2**(max_depth + 1) - 1
    thresholds = cppyy.gbl.std.vector[threshold_dtype](len_thresholds * num_trees)

    def fill_arrays(node, index, features_base, thresholds_base):
        # Set leaf score as threshold value if this node is a leaf
        if "leaf" in node:
            thresholds[thresholds_base + index] = node["leaf"]
            return

        # Set feature index
        feature = int(node["split"].replace("f", ""))
        features[features_base + index] = feature

        # Set threshold value
        thresholds[thresholds_base + index] = node["split_condition"]

        # Find next left (no) and right (yes) node
        if node["children"][0]["nodeid"] == node["yes"]:
            yes, no = 1, 0
        else:
            yes, no = 0, 1

        # Fill values from the child nodes
        fill_arrays(node["children"][no], 2 * index + 1, features_base, thresholds_base)
        fill_arrays(node["children"][yes], 2 * index + 2, features_base, thresholds_base)

    for i_tree, tree in enumerate(forest):
        fill_arrays(tree, 0, len_features * i_tree, len_thresholds * i_tree)

    # Store arrays in a ROOT file in a folder with the given key name
    # TODO: Write single values as simple integers and not vectors.
    f = cppyy.gbl.TFile(output_path, "RECREATE")
    f.mkdir(key_name)
    d = f.Get(key_name)
    d.WriteObjectAny(features, "std::vector<int>", "features")
    d.WriteObjectAny(thresholds, "std::vector<" + threshold_dtype + ">", "thresholds")
    d.WriteObjectAny(objective, "std::string", "objective")
    max_depth_ = cppyy.gbl.std.vector["int"](1, max_depth)
    d.WriteObjectAny(max_depth_, "std::vector<int>", "max_depth")
    num_trees_ = cppyy.gbl.std.vector["int"](1, num_trees)
    d.WriteObjectAny(num_trees_, "std::vector<int>", "num_trees")
    num_features_ = cppyy.gbl.std.vector["int"](1, num_features)
    d.WriteObjectAny(num_features_, "std::vector<int>", "num_features")
    num_classes_ = cppyy.gbl.std.vector["int"](1, num_classes)
    d.WriteObjectAny(num_classes_, "std::vector<int>", "num_classes")
    f.Write()
    f.Close()


@pythonization()
def pythonize_tree_inference(klass, name):
    # Parameters:
    # klass: class to be pythonized
    # name: string containing the name of the class

    if name == "TMVA::Experimental::SaveXGBoost":
        klass.__init__ = SaveXGBoost

    return True
