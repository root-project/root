### \file
### \ingroup tutorial_ml
### \notebook -nodraw
### Example of getting batches of events from a ROOT dataset as Python
### generators of numpy arrays.
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

import ROOT

tree_name = "sig_tree"
file_name = str(ROOT.gROOT.GetTutorialDir()) + "/machine_learning/data/Higgs_data.root"

batch_size = 128

rdataframe = ROOT.RDataFrame(tree_name, file_name)

target = "Type"

num_of_epochs = 2

dl = ROOT.Experimental.ML.RDataLoader(
    rdataframe,
    batch_size,
    target=target,
    shuffle=True,
    drop_remainder=True,
)

gen_train, gen_validation = dl.train_test_split(test_size=0.3)

for i in range(num_of_epochs):
    # Loop through training set
    for i, (x_train, y_train) in enumerate(gen_train.as_numpy()):
        print(f"Training batch {i + 1} => x: {x_train.shape}, y: {y_train.shape}")

    # Loop through Validation set
    for i, (x_validation, y_validation) in enumerate(gen_validation.as_numpy()):
        print(f"Validation batch {i + 1} => x: {x_validation.shape}, y: {y_validation.shape}")
