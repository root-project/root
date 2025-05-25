### \file
### \ingroup tutorial_ml
### \notebook -nodraw
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

##################################################
# This tutorial shows the usage of filters and vectors
# when using RBatchGenerator
##################################################

import ROOT


tree_name = "test_tree"
file_name = (
    ROOT.gROOT.GetTutorialDir().Data()
    + "/machine_learning/RBatchGenerator_filters_vectors_hvector.root"
)

chunk_size = 50  # Defines the size of the chunks
batch_size = 5  # Defines the size of the returned batches

rdataframe = ROOT.RDataFrame(tree_name, file_name)

# Define filters, filters must be named
filteredrdf = rdataframe.Filter("f1 > 30", "first_filter")\
                .Filter("f2 < 70", "second_filter")\
                .Filter("f3==true", "third_filter")

max_vec_sizes = {"f4": 3, "f5": 2, "f6": 1}

ds_train, ds_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    filteredrdf,
    batch_size,
    chunk_size,
    validation_split=0.3,
    max_vec_sizes=max_vec_sizes,
    shuffle=True,
)

print(f"Columns: {ds_train.columns}")

for i, b in enumerate(ds_train):
    print(f"Training batch {i} => {b.shape}")

for i, b in enumerate(ds_validation):
    print(f"Validation batch {i} => {b.shape}")
