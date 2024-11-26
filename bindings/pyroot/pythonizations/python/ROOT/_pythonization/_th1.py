# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
/**
\class TH1
\brief \parblock \endparblock
\htmlonly
<details open>
<summary  style="font-size:20px; color: #425788;"><b>Python interface</b></summary>
<div class="pyrootbox">
\endhtmlonly

## Fitting histograms in Python

One-dimensional histograms can be fit in [Python](https://root.cern/manual/python) with a similar syntax as in C++.
To fit a 1D histogram to one of the ROOT standard functions (e.g. a Gaussian):

\code{.py}
# Create and initialize a test histogram to fit
myTH1D = ROOT.TH1D("th1d", "Histogram for fitting", 200, 0, 10)
myTH1D.FillRandom("gaus", 1000)

# Fit to a ROOT pre-defined Gaussian function "gaus"
myTH1D.Fit("gaus")
\endcode

The list of standard functions in ROOT can be accessed with the TROOT::GetListOfFunctions.
In Python, the standard functions for TF1 can be printed as follows:

\code{.py}
ROOT.TF1.InitStandardFunctions()

# Print a list of available functions and their definitions
ROOT.gROOT.GetListOfFunctions().Print()
\endcode

## Accessing results of the fit in Python

To access the results of the fit, run the TH1::Fit method with the "s" option (please see the TH1::Fit(TF1*, Option_t*, Option_t*, Double_t, Double_t)
documentation for a list of possible options).
This will return a TFitResult which can be examined with the corresponding TFitResult methods, with the same names in Python as in C++.

For example:

\code{.py}
# Re-using the TH1D defined in the earlier example code
myResult = myTH1D.Fit("gaus", "s")

# Get the fitted parameters as a vector
myResult.Parameters()

# Get the error of the first parameter
myResult.ParError(0) 
\endcode


## Fitting to user-defined functions in Python
1D histograms can also be fit to any user-defined function expressed as a TF1 (see the TF1 documentation for examples on how to do this).

For example, a TF1 can be defined and initialized with its ROOT constructor:

\code{.py}
# Define the function, e.g. a polynomial with two parameters: y(x) = a * x^b
myTF1 = ROOT.TF1("myFunction", "[0] * pow(x, [1])", 0, 10)

# Set parameters
myTF1.SetParameters(10.0, 4.0)

# Initialize a test histogram to fit, and fit it
myTH1D = ROOT.TH1D("th1d", "My histogram to fit", 200, 0, 10)
myTH1D.FillRandom("myFunction", 1000)
myTH1D.Fit("myFunction")
\endcode

A TF1 can also be defined using a Python function, for example:

\code{.py}
def myGaussian(x, pars):
    '''
    Defines a Gaussian function
    '''
    return pars[0]*np.exp(-0.5* pow(x[0] - pars[1], 2)) 

# Initialize from the Python function with the range -5 to +5, with two parameters to fit, and a one-dimensional input x
myTF1 = ROOT.TF1("myFunction", myGaussian, -5, 5, npar=2, ndim=1) 

# Create a 1D histogram and initialize it with the built-in ROOT Gaussian "gaus"
myTH1D = ROOT.TH1D("th1d", "Test", 200, -5, 5)
myTH1D.FillRandom("gaus", 1000)

# Fit the 1D histogram to our custom Python function
myTH1D.Fit("myFunction")
\endcode

## Further Python fitting examples
Further examples can be found in the tutorials:
- [combinedFit.py](combinedFit__8py.html) performs a combined (simultaneous) fit of two 1D histograms with separate functions and some common parameters.
- [fit1.py](fit1__8py.html) reads a `TF1` and 1D histogram (created and saved in an earlier example [fillrandom.py](fillrandom__8py.html)), and fits the histogram.
- [fitConvolution.py](fitConvolution__8py.html) fits a 1D histogram to a convolution of two functions.
- [fitNormSum.py](fitNormSum__8py.html) fits a 1D histogram to the normalized sum of two functions (here, a background exponential and a crystal ball function).
- [multifit.py](multifit__8py.html) fits multiple functions to different ranges of a 1D histogram.


\htmlonly
</div>
</details>
\endhtmlonly
*/
"""

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
