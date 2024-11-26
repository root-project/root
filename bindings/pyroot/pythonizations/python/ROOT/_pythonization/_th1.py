# Author: Enric Tejedor CERN  02/2019

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

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

def _getbin(self, bin_or_callable):
    # print("getitem called with", index_or_callable)
    print(type(bin_or_callable))
    if callable(bin_or_callable):
        bin = bin_or_callable(self)
    elif isinstance(bin_or_callable, int):
        # bin 0 is underflow bin which is treated 
        # in a callable
        bin = bin_or_callable + 1
    else:
        raise TypeError(f"Expected 'bin_or_callable' to be of type int or callable, but got {type(bin_or_callable).__name__} instead.")
        
    print("getitem at", bin)
    return bin

def _getitem(self, bin_or_callable):
    bin = _getbin(self, bin_or_callable)
    return self.GetBinContent(bin)

def _setitem(self, bin_or_callable, value):
    bin = _getbin(self, bin_or_callable)
    print("type of bin ", type(bin))
    return self.SetBinContent(bin, value)


@pythonization('TH1')
def pythonize_th1(klass):
    # Parameters:
    # klass: class to be pythonized
    print("pythonize TH1")

    # Support hist *= scalar
    klass.__imul__ = _imul

    # UHI pythonizations
    klass.__getitem__ = _getitem
    klass.__setitem__ = _setitem
