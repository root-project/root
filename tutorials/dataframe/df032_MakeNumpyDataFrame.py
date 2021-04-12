## \file
## \ingroup tutorial_dataframe
## \notebook
## Read data from Numpy arrays into RDataFrame.
##
## \macro_code
## \macro_output
##
## \date March 2021
## \author Stefan Wunsch (KIT, CERN)

import ROOT
import numpy as np

# Let's create some data in numpy arrays
x = np.array([1, 2, 3], dtype=np.int32)
y = np.array([4, 5, 6], dtype=np.float64)

# Read the data with RDataFrame
# The column names in the RDataFrame are defined by the keys of the dictionary.
# Please note that only fundamental types (int, float, ...) are supported and
# the arrays must have the same length.
df = ROOT.RDF.MakeNumpyDataFrame({'x': x, 'y': y})

# You can now use the RDataFrame as usually, e.g. add a column ...
df = df.Define('z', 'x + y')

# ... or print the content
df.Display().Print()

# ... or save the data as a ROOT file
df.Snapshot('tree', 'df032_MakeNumpyDataFrame.root')
