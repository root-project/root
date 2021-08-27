# Authors:
# * Jonas Rembser 06/2021
# * Harshal Shende 06/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._utils import _dict_to_std_map, cpp_signature


class RooCategory(object):
    r"""Constructor of RooCategory takes a map as an argument also supports python dictionaries.
    For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Default bindings :
    mixState = ROOT.RooCategory("mixState", "B0/B0bar mixing state")
    mixState.defineType("mixed", -1)
    mixState.defineType("unmixed", 1)

    # With pythonization :
    mixState = ROOT.RooCategory("mixState", "B0/B0bar mixing state", {"mixed" : -1, "unmixed" : 1})
    \endcode
    """

    @cpp_signature("RooCategory(const char* name, const char* title, const std::map<std::string, int>& allowedStates);")
    def __init__(self, *args, **kwargs):
        r"""The RooCategory constructor is pythonized for converting python dict to std::map.
        The instances in the dict must correspond to the template argument in std::map of the constructor.
        """
        # Redefinition of `RooCategory` constructor for converting python dict to std::map.
        if len(args) > 2 and isinstance(args[2], dict):
            args = list(args)
            args[2] = _dict_to_std_map(args[2], {"std::string": "int"})

        self._init(*args, **kwargs)
