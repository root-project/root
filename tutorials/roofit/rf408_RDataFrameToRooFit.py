## \file
## \ingroup tutorial_roofit
## \notebook
## Fill RooDataSet/RooDataHist in RDataFrame.
##
## This tutorial shows how to fill RooFit data classes directly from RDataFrame.
## Using two small helpers, we tell RDataFrame where the data has to go.
##
## \macro_code
## \macro_output
##
## \date July 2021
## \author Harshal Shende, Stephan Hageboeck (C++ version)

import ROOT
import math


# Set up
# ------------------------

# We create an RDataFrame with two columns filled with 2 million random numbers.
d = ROOT.RDataFrame(2000000)
dd = d.Define("x", "gRandom->Uniform(-5.,  5.)").Define("y", "gRandom->Gaus(1., 3.)")


# We create RooFit variables that will represent the dataset.
x = ROOT.RooRealVar("x", "x", -5.0, 5.0)
y = ROOT.RooRealVar("y", "y", -50.0, 50.0)
x.setBins(10)
y.setBins(20)


# Booking the creation of RooDataSet / RooDataHist in RDataFrame
# ----------------------------------------------------------------

# Method 1:
# ---------
# We directly book the RooDataSetHelper action.
# We need to pass
# - the RDataFrame column types as template parameters
# - the constructor arguments for RooDataSet (they follow the same syntax as the usual RooDataSet constructors)
# - the column names that RDataFrame should fill into the dataset
#
# NOTE: RDataFrame columns are matched to RooFit variables by position, *not by name*!
rooDataSet = dd.Book(
    ROOT.std.move(ROOT.RooDataSetHelper("dataset", "Title of dataset", ROOT.RooArgSet(x, y))), ("x", "y")
)


# Method 2:
# ---------
# We first declare the RooDataHistHelper
rdhMaker = ROOT.RooDataHistHelper("dataset", "Title of dataset", ROOT.RooArgSet(x, y))

# Then, we move it into an RDataFrame action:
rooDataHist = dd.Book(ROOT.std.move(rdhMaker), ("x", "y"))


# Run it and inspect the results
# -------------------------------

# Let's inspect the dataset / datahist.
# Note that the first time we touch one of those objects, the RDataFrame event loop will run.


def printData(data):
    print("")
    data.Print()
    for i in range(min(data.numEntries(), 20)):
        print(
            "("
            + ", ".join(["{0:8.3f}".format(var.getVal()) for var in data.get(i)])
            + ", )  weight={0:10.3f}".format(data.weight())
        )

    print("mean(x) = {0:.3f}".format(data.mean(x)) + "\tsigma(x) = {0:.3f}".format(math.sqrt(data.moment(x, 2.0))))
    print("mean(y) = {0:.3f}".format(data.mean(y)) + "\tsigma(y) = {0:.3f}\n".format(math.sqrt(data.moment(y, 2.0))))


printData(rooDataSet)
printData(rooDataHist)
