# Authors:
# * Jonas Rembser 05/2021
# * Harshal Shende 06/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._utils import _kwargs_to_roocmdargs, cpp_signature
from libcppyy import SetOwnership


class RooAbsCollection(object):
    r"""Some member functions of RooAbsCollection that take a RooCmdArg as argument also support keyword arguments.
    So far, this applies to RooAbsCollection::printLatex. For example, the following code is equivalent in PyROOT:
    \code{.py}
    # Directly passing a RooCmdArg:
    params.printLatex(ROOT.RooFit.Sibling(initParams), ROOT.RooFit.Columns(2))

    # With keyword arguments:
    params.printLatex(Sibling=initParams, Columns =2)
    \endcode
    """

    @cpp_signature("RooAbsArg *RooAbsCollection::addClone(const RooAbsArg& var, Bool_t silent=kFALSE) ;")
    def addClone(self, arg, silent=False):
        r"""The RooAbsCollection::addClone() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        clonedArg = self._addClone(arg, silent)
        SetOwnership(clonedArg, False)

    @cpp_signature("Bool_t RooAbsCollection::addOwned(RooAbsArg& var, Bool_t silent=kFALSE);")
    def addOwned(self, arg, silent=False):
        r"""The RooAbsCollection::addOwned() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function."""
        self._addOwned(arg, silent)
        SetOwnership(arg, False)

    @cpp_signature(
        "RooAbsCollection::printLatex(const RooCmdArg& arg1=RooCmdArg(), const RooCmdArg& arg2=RooCmdArg(),"
        "                        const RooCmdArg& arg3=RooCmdArg(), const RooCmdArg& arg4=RooCmdArg(),"
        "                        const RooCmdArg& arg5=RooCmdArg(), const RooCmdArg& arg6=RooCmdArg(),"
        "                        const RooCmdArg& arg7=RooCmdArg(), const RooCmdArg& arg8=RooCmdArg()) const ;"
    )
    def printLatex(self, *args, **kwargs):
        r"""The RooAbsCollection::printLatex() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsCollection.printLatex` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._printLatex(*args, **kwargs)
