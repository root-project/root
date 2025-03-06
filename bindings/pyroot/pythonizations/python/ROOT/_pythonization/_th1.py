# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

r"""
\pythondoc TH1

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

## Pythonizations
The TH1 class has several additions for its use from Python, which are also available in its subclasses (e.g., TH1F, TH1D).

### In-Place Multiplication

TH1 instances support in-place multiplication with a scalar value using the `*=` operator:

\code{.py}
import ROOT

h = ROOT.TH1D("h", "h", 100, -10, 10)
h.FillRandom("gaus", 1000)

# Multiply histogram contents by 2
h *= 2
\endcode

This operation is equivalent to calling `h.Scale(2)`.

### Filling with NumPy Arrays

The Fill method has been pythonized to accept NumPy arrays as input. This allows for efficient filling of histograms with large datasets:

\code{.py}
import ROOT
import numpy as np

# Create a histogram
h = ROOT.TH1D("h", "h", 100, -10, 10)

# Create sample data
data = np.random.normal(0, 2, 10000)

# Fill histogram with data
h.Fill(data)

# Fill with weights
weights = np.ones_like(data) * 0.5
h.Fill(data, weights)
\endcode

The Fill method accepts the following arguments when used with NumPy arrays:
- First argument: NumPy array containing the data to fill
- Second argument (optional): NumPy array containing the weights for each entry

<em>Please note</em> that when providing weights, the length of the weights array must match the length of the data array. If weights are not provided, all entries will have a weight of 1. A ValueError will be raised if the lengths don't match:

\code{.py}
# This will raise ValueError
data = np.array([1.0, 2.0, 3.0])
weights = np.array([0.5, 1.0])  # Wrong length!
h.Fill(data, weights)  # Raises ValueError: "Length mismatch: data length (3) != weights length (2)"
\endcode

The original Fill method functionality is preserved for non-NumPy arguments:

\code{.py}
# Traditional filling still works
h.Fill(1.0)  # Fill single value
h.Fill(1.0, 2.0)  # Fill single value with weight
\endcode

## Further Python fitting examples
Further examples can be found in the tutorials:
- [combinedFit.py](combinedFit_8py.html) performs a combined (simultaneous) fit of two 1D histograms with separate functions and some common parameters.
- [fit1.py](fit1_8py.html) reads a `TF1` and 1D histogram (created and saved in an earlier example [fillrandom.py](fillrandom__8py.html)), and fits the histogram.
- [fitConvolution.py](fitConvolution_8py.html) fits a 1D histogram to a convolution of two functions.
- [fitNormSum.py](fitNormSum_8py.html) fits a 1D histogram to the normalized sum of two functions (here, a background exponential and a crystal ball function).
- [multifit.py](multifit_8py.html) fits multiple functions to different ranges of a 1D histogram.

\endpythondoc
"""

from . import pythonization
from ROOT._pythonization._memory_utils import inject_constructor_releasing_ownership, inject_clone_releasing_ownership, _SetDirectory_SetOwnership

# Multiplication by constant

def _imul(self, c):
    # Parameters:
    # - self: histogram
    # - c: constant by which to multiply the histogram
    # Returns:
    # - A multiplied histogram (in place)
    self.Scale(c)
    return self

# Fill with numpy array

def _FillWithNumpyArray(self, *args):
    """
    Fill histogram with numpy array.
    Parameters:
    - self: histogram
    - args: arguments to FillN
            If the first argument is numpy.ndarray:
            - fills the histogram with this array
            - optional second argument is weights array,
              if not provided, weights of 1 are used
            Otherwise:
            - Arguments are passed directly to the original FillN method
    Returns:
    - Result of FillN if numpy case is detected, otherwise result of Fill
    Raises:
    - ValueError: If weights length doesn't match data length
    """
    import numpy as np

    if args and isinstance(args[0], np.ndarray):
        data = args[0]
        weights = np.ones(len(data)) if len(args) < 2 or args[1] is None else args[1]
        if len(weights) != len(data):
            raise ValueError(
                f"Length mismatch: data length ({len(data)}) != weights length ({len(weights)})"
            )
        return self.FillN(len(data), data, weights)
    else:
        return self._Fill(*args)


# The constructors need to be pythonized for each derived class separately:
_th1_derived_classes_to_pythonize = [
    "TH1C",
    "TH1S",
    "TH1I",
    "TH1L",
    "TH1F",
    "TH1D",
    "TH1K",
    "TProfile",
]

for klass in _th1_derived_classes_to_pythonize:
    pythonization(klass)(inject_constructor_releasing_ownership)

    from ROOT._pythonization._uhi import add_plotting_features
    
    # Add UHI components
    uhi_components = [add_plotting_features]
    for uc in uhi_components:
        pythonization(klass)(uc)

@pythonization('TH1')
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized

    # Support hist *= scalar
    klass.__imul__ = _imul

    # Support hist.Fill(numpy_array) and hist.Fill(numpy_array, numpy_array)
    klass._Fill = klass.Fill
    klass.Fill = _FillWithNumpyArray

    klass._Original_SetDirectory = klass.SetDirectory
    klass.SetDirectory = _SetDirectory_SetOwnership

    inject_clone_releasing_ownership(klass)
