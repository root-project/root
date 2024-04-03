### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
###
### Example of getting batches of events from a ROOT dataset as Python
### generators of numpy arrays.
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

import ROOT

tree_name = "sig_tree"
file_name = "http://root.cern/files/Higgs_data.root"

rdf = ROOT.RDataFrame(tree_name, file_name)

batch_size = 128
chunk_size = 5_000

ds_train, ds_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    batch_size,
    chunk_size,
    rdataframe=rdf,
    validation_split=0.3,
    shuffle=True,
)

# Loop through training set
i = 1
while True:
    try:
        b = next(ds_train)
        print(f"Training batch {i} => {b.shape}")
        i+=1
    except StopIteration:
        break


# Loop through Validation set
i = 1
while True:
    try:
        b = next(ds_validation)
        print(f"Validation batch {i} => {b.shape}")
        i+=1
    except StopIteration:
        break
