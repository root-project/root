### \file
### \ingroup tutorial_tmva
### \notebook -nodraw
### Example of getting batches of events from a ROOT dataset as Python
### generators of numpy arrays.
###
### \macro_code
### \macro_output
### \author Dante Niewenhuis

import ROOT

tree_name = "sig_tree"
file_name = "http://root.cern/files/Higgs_data.root"

batch_size = 128
chunk_size = 5_000

rdataframe = ROOT.RDataFrame(tree_name, file_name)

gen_train, gen_validation = ROOT.TMVA.Experimental.CreateNumPyGenerators(
    rdataframe,
    batch_size,
    chunk_size,
    validation_split=0.3,
    shuffle=True,
    drop_remainder=False
)

# Loop through training set
for i, b in enumerate(gen_train):
    print(f"Training batch {i} => {b.shape}")


# Loop through Validation set
for i, b in enumerate(gen_validation):
    print(f"Validation batch {i} => {b.shape}")
