### \file
### \ingroup tutorial_ml
### \notebook -nodraw
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

##################################################
# This tutorial shows the usage of filters and vectors
# when using the ROOT ML dataloader
##################################################

import ROOT

tree_name = "test_tree"
file_name = ROOT.gROOT.GetTutorialDir().Data() + "/machine_learning/ml_dataloader_filters_vectors_hvector.root"

batch_size = 5  # Defines the size of the returned batches

rdataframe = ROOT.RDataFrame(tree_name, file_name)

# Define filters, filters must be named
filteredrdf = (
    rdataframe.Filter("f1 > 30", "first_filter").Filter("f2 < 70", "second_filter").Filter("f3==true", "third_filter")
)

max_vec_sizes = {"f4": 3, "f5": 2, "f6": 1}

dl = ROOT.Experimental.ML.RDataLoader(
    filteredrdf,
    batch_size,
    max_vec_sizes=max_vec_sizes,
    shuffle=False,
)

ds_train, ds_validation = dl.train_test_split(test_size=0.3)

print(f"Columns: {ds_train.columns}")

for i, b in enumerate(ds_train.as_numpy()):
    print(f"Training batch {i} => {b.shape}")

for i, b in enumerate(ds_validation.as_numpy()):
    print(f"Validation batch {i} => {b.shape}")
