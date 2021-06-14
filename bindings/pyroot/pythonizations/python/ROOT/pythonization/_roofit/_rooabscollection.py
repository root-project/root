# Authors:
# * Jonas Rembser 05/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from libcppyy import SetOwnership


class RooAbsCollection(object):
    def addOwned(self, arg, silent=False):
        self._addOwned(arg, silent)
        SetOwnership(arg, False)
