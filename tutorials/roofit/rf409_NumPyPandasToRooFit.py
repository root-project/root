## \file
## \ingroup tutorial_roofit
## \notebook
## Convert between NumPy arrays or Pandas DataFrames and RooDataSets.
##
## This totorial first how to export a RooDataSet to NumPy arrays or a Pandas
## DataFrame, and then it shows you how to create a RooDataSet from a Pandas
## DataFrame.
##
## \macro_code
## \macro_output
##
## \date November 2021
## \author Jonas Rembser

import ROOT

import numpy as np


# The number of events that we use for the datasets created in this tutorial.
n_events = 10000


# Creating a RooDataSet and exporting it to the Python ecosystem
# --------------------------------------------------------------

# Define the observable.
x = ROOT.RooRealVar("x", "x", -10, 10)

# Define a Gaussian model with its parameters.
mean = ROOT.RooRealVar("mean", "mean of gaussian", 1, -10, 10)
sigma = ROOT.RooRealVar("sigma", "width of gaussian", 1, 0.1, 10)
gauss = ROOT.RooGaussian("gauss", "gaussian PDF", x, mean, sigma)

# Create a RooDataSet.
data = gauss.generate(ROOT.RooArgSet(x), 10000)

# Use RooDataSet.to_numpy() to export dataset to a dictionary of NumPy arrays.
# Real values will be of type `double`, categorical values of type `int`.
arrays = data.to_numpy()

# We can verify that the mean and standard deviation matches our model specification.
print("Mean of numpy array:", np.mean(arrays["x"]))
print("Standard deviation of numpy array:", np.std(arrays["x"]))

# It is also possible to create a Pandas DataFrame directly from the numpy arrays:
df = data.to_pandas()

# Now you can use the DataFrame e.g. for plotting. You can even combine this
# with the RooAbsReal.bins PyROOT function, which returns the binning from
# RooFit as a numpy array!
try:
    import matplotlib.pyplot as plt

    df.hist(column="x", bins=x.bins())
except Exception:
    print(
        'Skipping `df.hist(column="x", bins=x.bins())` because matplotlib could not be imported or was not able to display the plot.'
    )

del data
del arrays
del df


# Creating a dataset with NumPy and importing it to a RooDataSet
# --------------------------------------------------------------

# Now we create some Gaussian toy data with numpy, this time with a different
# mean.
x_arr = np.random.normal(-1.0, 1.0, (n_events,))

# Import the data to a RooDataSet, passing a dictionary of arrays and the
# corresponding RooRealVars just like you would pass to the RooDataSet
# constructor.
data = ROOT.RooDataSet.from_numpy({"x": x_arr}, [x])

# Let's fit the Gaussian to the data. The mean is updated accordingly.
fit_result = gauss.fitTo(data, PrintLevel=-1, Save=True)
fit_result.Print()

# We can now plot the model and the dataset with RooFit.
xframe = x.frame(Title="Gaussian pdf")
data.plotOn(xframe)
gauss.plotOn(xframe)

# Draw RooFit plot on a canvas.
c = ROOT.TCanvas("rf409_NumPyPandasToRooFit", "rf409_NumPyPandasToRooFit", 800, 400)
xframe.Draw()
c.SaveAs("rf409_NumPyPandasToRooFit.png")


# Exporting a RooDataHist to NumPy arrays for histogram counts and bin edges
# --------------------------------------------------------------------------

def print_histogram_output(histogram_output):
    counts, bin_edges = histogram_output
    print(np.array(counts, dtype=int))
    print(bin_edges[0])


# Create a binned clone of the dataset to show RooDataHist to NumPy export.
datahist = data.binnedClone()

# You can also export a RooDataHist to numpy arrays with
# RooDataHist.to_numpy(). As output, you will get a multidimensional array with
# the histogram counts and a list of arrays with bin edges. This is comparable
# to the ouput of numpy.histogram (or numpy.histogramdd for the
# multidimensional case).
counts, bin_edges = datahist.to_numpy()

print("Counts and bin edges from RooDataHist.to_numpy:")
print_histogram_output((counts, bin_edges))

# Let's compare the ouput to the counts and bin edges we get with
# numpy.histogramdd when we pass it the original samples:
print("Counts and bin edges from np.histogram:")
print_histogram_output(np.histogramdd([x_arr], bins=[x.bins()]))

# The array values should be the same!


# Importing a RooDataHist from NumPy arrays with histogram counts and bin edges
# -----------------------------------------------------------------------------

# There is also a `RooDataHist.from_numpy` function, again with an interface
# inspired by `numpy.histogramdd`. You need to pass at least the histogram
# counts and the list of variables. The binning is optional: the default
# binning of the RooRealVars is used if not explicitly specified.
datahist_new_1 = ROOT.RooDataHist.from_numpy(counts, [x])

print("RooDataHist imported with default binning and exported back to numpy:")
print_histogram_output(datahist_new_1.to_numpy())


# It's also possible to pass custom bin edges to `RooDataHist.from_numpy`, just
# like you pass them to `numpy.histogramdd` when you get the counts to fill the
# RooDataHist with:
bins = [np.linspace(-10, 10, 21)]
counts, _ = np.histogramdd([x_arr], bins=bins)
datahist_new_2 = ROOT.RooDataHist.from_numpy(counts, [x], bins=bins)

print("RooDataHist imported with linspace binning and exported back to numpy:")
print_histogram_output(datahist_new_2.to_numpy())

# Alternatively, you can specify only the number of bins and the range if your
# binning is uniform. This is preferred over passing the full list of bin
# edges, because RooFit will know that the binning is uniform and do some
# optimizations.
bins = [20]
ranges = [(-10, 10)]
counts, _ = np.histogramdd([x_arr], bins=bins, range=ranges)
datahist_new_3 = ROOT.RooDataHist.from_numpy(counts, [x], bins=bins, ranges=ranges)

print("RooDataHist imported with uniform binning and exported back to numpy:")
print_histogram_output(datahist_new_3.to_numpy())
