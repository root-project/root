##################################################
# This tutorial shows the usage of filters and vectors
# when using RBatchGenerator
##################################################

import ROOT


tree_name = "test_tree"
file_name = (
    ROOT.gROOT.GetTutorialDir().Data()
    + "/tmva/RBatchGenerator_filters_vectors_hvector.root"
)

chunk_size = 50  # Defines the size of the chunks
batch_size = 5  # Defines the size of the returned batches

# Define filters as strings
filters = ["f1 > 30", "f2 < 70", "f3 == true"]
max_vec_sizes = {"f4": 3, "f5": 2, "f6": 1}

ds_train, ds_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    tree_name,
    file_name,
    batch_size,
    chunk_size,
    validation_split=0.3,
    filters=filters,
    max_vec_sizes=max_vec_sizes,
    shuffle=True,
)

print(f"Columns: {ds_train.columns}")

for i, b in enumerate(ds_train):
    print(f"Training batch {i} => {b.shape}")

for i, b in enumerate(ds_validation):
    print(f"Validation batch {i} => {b.shape}")
