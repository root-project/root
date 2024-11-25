# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r'''
/**
\class ROOT::TH1
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly
\anchor python


## Fitting histograms in PyROOT
One-dimensional histograms can be fit in [PyROOT](https://root.cern/manual/python) with a similar syntax as in C++. To fit a 1D histogram to one of the ROOT standard functions (e.g Gaussian):
~~~{.py}
# Create and initialize a test histogram to fit
myTH1D = ROOT.TH1D("th1d", "Histogram for fitting", 200, 0, 10)
myTH1D.FillRandom("gaus", 1000)
# Fit to a ROOT pre-defined Gaussian function "gaus"
myTH1D.Fit("gaus")
~~~

The list of standard functions in ROOT can be accessed with the [gROOT](classTROOT.html) `GetListOfFunctions()`:
~~~{.py}
ROOT.TF1.InitStandardFunctions()
# Print a list of available functions and their definitions
ROOT.gROOT.GetListOfFunctions().Print()
~~~

## Accessing results of the fit in PyROOT
To access the results of the fit, run the `.Fit()` method with the `s` option which returns a [TFitResult](classTFitResult.html). All other [TFitResult](classTFitResult.html) methods can be called with same name as the C++ method. Continuing the short example from before:
~~~{.py}
# Call .Fit() with the "s" method to save results
myResult = myTH1D.Fit("gaus", "s")

# Get the fitted parameters as a vector
myResult.Parameters()

# Get the error of the first parameter
myResult.ParError(0) 
~~~
For a full list and description of options to `.Fit()` (valid in both C++ and PyROOT) please see the [.Fit() method documentation](classTH1.html).

## Fitting to user-defined functions in PyROOT
1D histograms can also be fit to any user-defined `TF1` (see the `TF1` documentation). The `TF1` can be defined in a ROOT constructor, for example:
~~~{.py}
# Define the function, e.g. a polynomial with two parameters: y(x) = a * x^b
myTF1 = ROOT.TF1("myFunction", "[0] * pow(x, [1])", 0, 10)

# Set parameters
myTF1.SetParameters(10.0, 4.0)

# Initialize a test histogram to fit, and fit it
myTH1D = ROOT.TH1D("th1d", "My histogram to fit", 200, 0, 10)
myTH1D.FillRandom("myFunction", 1000)
myTH1D.Fit("myFunction")
~~~

The `TF1` function can also be initialized from a Python function (see the `TF1` documentation), for example:
~~~{.py}
def myGaussian(x, pars):
    """
    Defines a Gaussian function
    """
    return pars[0]*np.exp(-0.5* pow(x[0] - pars[1], 2)) 

# Initialize from the Python function with the range -5 to +5, with two parameters to fit, and a one-dimensional input x
myTF1 = ROOT.TF1("myFunction", myGaussian, -5, 5, npar=2, ndim=1) 

# Create a 1D histogram and initialize it with the built-in ROOT Gaussian "gaus"
myTH1D = ROOT.TH1D("th1d", "Test", 200, -5, 5)
myTH1D.FillRandom("gaus", 1000)

# Fit the 1D histogram to our custom Python function
myTH1D.Fit("myFunction")
~~~

## Further PyROOT fitting examples
Further examples can be found in the tutorials:
- [combinedFit.py](combinedFit__8py.html) performs a combined (simultaneous) fit of two 1D histograms with separate functions and some common parameters.
- [fit1.py](fit1__8py.html) reads a `TF1` and 1D histogram (created and saved in an earlier example [fillrandom.py](fillrandom__8py.html)), and fits the histogram.
- [fitConvolution.py](fitConvolution__8py.html) fits a 1D histogram to a convolution of two functions.
- [fitNormSum.py](fitNormSum__8py.html) fits a 1D histogram to the normalized sum of two functions (here, a background exponential and a crystal ball function).
- [multifit.py](multifit__8py.html) fits multiple functions to different ranges of a 1D histogram.

\htmlonly
</div>
\endhtmlonly

\anchor reference
*/
'''

from . import pythonization


# Multiplication by constant

def _imul(self, c):
    # Parameters:
    # - self: histogram
    # - c: constant by which to multiply the histogram
    # Returns:
    # - A multiplied histogram (in place)
    self.Scale(c)
    return self


@pythonization('TH1')
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized

    # Support hist *= scalar
    klass.__imul__ = _imul
