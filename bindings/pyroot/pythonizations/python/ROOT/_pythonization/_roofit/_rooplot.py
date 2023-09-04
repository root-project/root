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
    def addObject(self, *args, **kwargs):
        from ROOT._pythonization._memory_utils import declare_cpp_owned_arg

        # Python should transfer the ownership to the RooPlot
        declare_cpp_owned_arg(0, "obj", args, kwargs)

        return self._addObject(*args, **kwargs)
