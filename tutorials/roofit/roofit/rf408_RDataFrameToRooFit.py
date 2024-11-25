## \file
## \ingroup tutorial_roofit_main
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
df = ROOT.RDataFrame(2000000).Define("x", "gRandom->Uniform(-5.,  5.)").Define("y", "gRandom->Gaus(1., 3.)")


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

# NOTE: RDataFrame columns are matched to RooFit variables by position, *not by name*!
#
# The returned object is not yet a RooDataSet, but an RResultPtr that will be
# lazy-evaluated once you call GetValue() on it. We will only evaluate the
# RResultPtr once all other RDataFrame related actions are declared. This way
# we trigger the event loop computation only once, which will improve the
# runtime significantly.
#
# To learn more about lazy actions, see:
#     https://root.cern/doc/master/classROOT_1_1RDataFrame.html#actions
roo_data_set_result = df.Book(
    ROOT.std.move(ROOT.RooDataSetHelper("dataset", "Title of dataset", ROOT.RooArgSet(x, y))), ("x", "y")
)

# Method 2:
# ---------
# We first declare the RooDataHistHelper
rdhMaker = ROOT.RooDataHistHelper("dataset", "Title of dataset", ROOT.RooArgSet(x, y))

# Then, we move it into an RDataFrame action:
roo_data_hist_result = df.Book(ROOT.std.move(rdhMaker), ("x", "y"))


# Run it and inspect the results
# -------------------------------

# At this point, all RDF actions were defined (namely, the `Book` operations),
# so we can get values from the RResultPtr objects, triggering the event loop
# and getting the actual RooFit data objects.
roo_data_set = roo_data_set_result.GetValue()
roo_data_hist = roo_data_hist_result.GetValue()

# Let's inspect the dataset / datahist.


def print_data(data):
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


print_data(roo_data_set)
print_data(roo_data_hist)
