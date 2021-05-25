## \file
## \ingroup tutorial_pyroot
## \notebook -nodraw
## This tutorial shows how a TTree can be quickly converted to a numpy array or
## a pandas.DataFrame.
##
## \macro_code
## \macro_output
##
## \date April 2018
## \author Stefan Wunsch

import ROOT
from sys import exit

try:
    import numpy as np
except:
    print("Failed to import numpy.")
    exit()


# Helper function to create an example tree
def make_example():
    root_file = ROOT.TFile("pyroot002_example.root", "RECREATE")
    tree = ROOT.TTree("tree", "tutorial")
    x = np.empty((1), dtype="float32")
    y = np.empty((1), dtype="float32")
    tree.Branch("x", x, "x/F")
    tree.Branch("y", y, "y/F")

    for i in range(4):
        x[0] = i
        y[0] = -i
        tree.Fill()
    root_file.Write()

    return (root_file, x, y), tree


# The conversion of the TTree to a numpy array is implemented with multi-
# thread support.
ROOT.ROOT.EnableImplicitMT()

# Create a ROOT file with a tree and the branches "x" and "y"
_, tree = make_example()

# Print content of the tree by looping explicitly
print("Tree content:\n{}\n".format(
    np.asarray([[tree.x, tree.y] for event in tree])))

# Read-out full tree as numpy array
array = tree.AsMatrix()
print("Tree converted to a numpy array:\n{}\n".format(array))

# Get numpy array and according labels of the columns
array, labels = tree.AsMatrix(return_labels=True)
print("Return numpy array and labels:\n{}\n{}\n".format(labels, array))

# Apply numpy methods on the data
print("Mean of the columns retrieved with a numpy method: {}\n".format(
    np.mean(array, axis=0)))

# Read only specific branches
array = tree.AsMatrix(columns=["x"])
print("Only the content of the branch 'x':\n{}\n".format(np.squeeze(array)))

array = tree.AsMatrix(exclude=["x"])
print("Read all branches except 'x':\n{}\n".format(np.squeeze(array)))

# Get an array with a specific data-type
array = tree.AsMatrix(dtype="int")
print("Return numpy array with data-type 'int':\n{}\n".format(array))

## Convert the tree to a pandas.DataFrame
try:
    import pandas
except:
    print("Failed to import pandas.")
    exit()

data, columns = tree.AsMatrix(return_labels=True)
df = pandas.DataFrame(data=data, columns=columns)
print("Tree converted to a pandas.DataFrame:\n{}".format(df))
