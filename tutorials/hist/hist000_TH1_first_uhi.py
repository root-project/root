## \file
## \ingroup tutorial_hist
## Hello World example for TH1
##
## Shows how to create, fill and write a histogram to a ROOT file.
##
## \macro_code
##
## \date July 2025
## \author Giacomo Parolini, Nursena Bitirgen

import numpy as np
import ROOT

# Open the file to write the histogram to
with ROOT.TFile.Open("outfile_uhi.root", "RECREATE") as outFile:
    # Create the histogram object
    # There are several constructors you can use (see TH1). In this example we use the
    # simplest one, accepting a number of bins and a range.
    histogram = ROOT.TH1D("histogram", "My first ROOT histogram", nbinsx=30, xlow=0.0, xup=10.0)

    # Fill the histogram by passing a NumPy array. In this simple example we use a fake set of data.
    # The 'D' in TH1D stands for 'double', so we fill the histogram with doubles.
    # In general you should prefer TH1D over TH1F unless you have a very specific reason
    # to do otherwise.
    values = np.array([1, 2, 3, 3, 3, 4, 3, 2, 1, 0])
    counts, edges = np.histogram(values, bins=30, range=(0.0, 10.0))
    histogram[...] = counts

    # Write the histogram to `outFile`.
    outFile.WriteObject(histogram, histogram.GetName())

    # When `with` block exits, `outFile` will close itself and write its contents to disk.
