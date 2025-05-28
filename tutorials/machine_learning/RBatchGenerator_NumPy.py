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
chunk_size = 5000
block_size = 400

rdataframe = ROOT.RDataFrame(tree_name, file_name)

target = "Type"

num_of_epochs = 2

gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    rdataframe,
    batch_size,    
    chunk_size,
    block_size,        
    target = target,
    validation_split = 0.3,
    shuffle = True,
    drop_remainder = True
)

for i in range(num_of_epochs):
    # Loop through training set
    for i, (x_train, y_train) in enumerate(gen_train):
        print(f"Training batch {i + 1} => x: {x_train.shape}, y: {y_train.shape}")

    # Loop through Validation set
    for i, (x_validation, y_validation) in enumerate(gen_validation):
        print(f"Validation batch {i + 1} => x: {x_validation.shape}, y: {y_validation.shape}")
