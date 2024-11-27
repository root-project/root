# Author: Lukas Breitwieser, CERN, 11/2024

################################################################################
# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

class loc:
    def __init__(self, value):
        self.value = value
        self.offset = 0 

    def __add__(self, add_value):
        self.offset = self.offset + add_value
        return self

    def __sub__(self, sub_value):
        self.offset = self.offset - sub_value
        return self

    def __call__(self, histogram):
        return histogram.FindBin(self.value) + self.offset

def underflow(histogram):
    return 0

def overflow(histogram):
    return histogram.GetNcells() - 1

class rebin:
    def __init__(self, ngroup, new_name = ""):
        self.ngroup = ngroup
        self.new_name = new_name

    def __call__(self, histogram):
        return histogram.Rebin(self.ngroup, self.new_name)

class sum:
    def __call__(self, histogram):
        return histogram.Integral()

"""
    Helper functions used by the facade to add UHI related symbols 
    to the module
"""
def add_uhi_helper(module):
    module.loc = loc
    module.underflow = underflow
    module.overflow = overflow
    module.rebin = rebin
    module.sum = sum

