# Authors:
# * Jonas Rembser 09/2023

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


class RooPlot(object):
    def addObject(self, obj, *args, **kwargs):
        import ROOT

        # PyROOT transfers the ownership to the RooPlot.
        ROOT.SetOwnership(obj, False)
        return self._addObject(obj, *args, **kwargs)
