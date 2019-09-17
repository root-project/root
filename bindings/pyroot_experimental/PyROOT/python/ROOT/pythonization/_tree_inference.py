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
    num_inputs = xgb_model._features_count
    objective_map = {
            "multi:softprob": "softmax", # Naming the objective softmax is more common today
            "binary:logistic": "logistic",
            }
    model_objective = xgb_model.objective
    num_outputs = xgb_model.n_classes_
    if "binary:" in model_objective:
        if num_outputs == 2:
            num_outputs = 1
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
    len_inputs = 2**max_depth - 1
    inputs = cppyy.gbl.std.vector["int"](len_inputs * num_trees, -1)
    len_thresholds = 2**(max_depth + 1) - 1
    thresholds = cppyy.gbl.std.vector[threshold_dtype](len_thresholds * num_trees)

    def fill_arrays(node, index, inputs_base, thresholds_base):
        # Set leaf score as threshold value if this node is a leaf
        if "leaf" in node:
            thresholds[thresholds_base + index] = node["leaf"]
            return

        # Set input index
        input_ = int(node["split"].replace("f", ""))
        inputs[inputs_base + index] = input_

        # Set threshold value
        thresholds[thresholds_base + index] = node["split_condition"]

        # Find next left (no) and right (yes) node
        if node["children"][0]["nodeid"] == node["yes"]:
            yes, no = 1, 0
        else:
            yes, no = 0, 1

        # Fill values from the child nodes
        fill_arrays(node["children"][no], 2 * index + 1, inputs_base, thresholds_base)
        fill_arrays(node["children"][yes], 2 * index + 2, inputs_base, thresholds_base)

    for i_tree, tree in enumerate(forest):
        fill_arrays(tree, 0, len_inputs * i_tree, len_thresholds * i_tree)

    # Determine to which output node a tree belongs
    outputs = cppyy.gbl.std.vector["int"](num_trees)
    if num_outputs != 1:
        for i in range(num_trees):
            outputs[i] = int(i % num_outputs)

    # Store arrays in a ROOT file in a folder with the given key name
    # TODO: Write single values as simple integers and not vectors.
    f = cppyy.gbl.TFile(output_path, "RECREATE")
    f.mkdir(key_name)
    d = f.Get(key_name)
    d.WriteObjectAny(inputs, "std::vector<int>", "inputs")
    d.WriteObjectAny(outputs, "std::vector<int>", "outputs")
    d.WriteObjectAny(thresholds, "std::vector<" + threshold_dtype + ">", "thresholds")
    d.WriteObjectAny(objective, "std::string", "objective")
    max_depth_ = cppyy.gbl.std.vector["int"](1, max_depth)
    d.WriteObjectAny(max_depth_, "std::vector<int>", "max_depth")
    num_trees_ = cppyy.gbl.std.vector["int"](1, num_trees)
    d.WriteObjectAny(num_trees_, "std::vector<int>", "num_trees")
    num_inputs_ = cppyy.gbl.std.vector["int"](1, num_inputs)
    d.WriteObjectAny(num_inputs_, "std::vector<int>", "num_inputs")
    num_outputs_ = cppyy.gbl.std.vector["int"](1, num_outputs)
    d.WriteObjectAny(num_outputs_, "std::vector<int>", "num_outputs")
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
