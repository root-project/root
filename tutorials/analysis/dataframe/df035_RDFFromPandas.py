## \file
## \ingroup tutorial_dataframe
## \notebook
## Read data from Pandas Data Frame into RDataFrame.
##
## \macro_code
## \macro_output
##
## \date February 2024
## \author Pere Mato (CERN)

import ROOT
import pandas as pd

# Let's create some data in a pandas dataframe
pdf = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})

# Convert the Pandas DataFrame to RDataFrame
# The column names are directly copied to the RDF 
# Please note that only fundamental types (int, float, ...) are supported and
# the arrays must have the same length.
df = ROOT.RDF.FromPandas(pdf)

# You can now use the RDataFrame as usually, e.g. add a column ...
df = df.Define('z', 'x + y')

# ... or print the content
df.Display().Print()

# ... or save the data as a ROOT file
df.Snapshot('tree', 'df035_RDFFromPandas.root')
