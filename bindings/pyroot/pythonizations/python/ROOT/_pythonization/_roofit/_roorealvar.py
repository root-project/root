# Authors:
# * Jonas Rembser 11/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


class RooRealVar(object):
    def bins(self, range_name=None):
        """Return the binning of this RooRealVar as a numpy array."""

        import numpy as np

        if range_name:
            binning = self.getBinning(range_name)
        else:
            binning = self.getBinning()

        num_bins = binning.numBins()
        bin_array = np.zeros(num_bins + 1)
        if num_bins == 0:
            return bin_array

        bin_array[0] = binning.binLow(0)
        bin_array[1] = binning.binHigh(0)
        for i in range(1, num_bins):
            a_min, a_max = binning.binLow(i), binning.binHigh(i)
            try:
                np.testing.assert_almost_equal(a_min, bin_array[i])
            except AssertionError:
                raise ValueError("Binnings with gaps in between can't be exported to numpy.")
            bin_array[i + 1] = a_max

        return bin_array
