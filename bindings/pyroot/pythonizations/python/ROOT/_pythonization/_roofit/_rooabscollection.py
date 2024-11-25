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
    __cpp_name__ = 'RooAbsCollection'

    def addClone(self, arg, silent=False):

        import ROOT

        clonedArg = self._addClone(arg, silent)
        # There are two overloads of RooAbsCollection::addClone():
        #
        #   - RooAbsArg *addClone(const RooAbsArg& var, bool silent=false);
        #   - void addClone(const RooAbsCollection& list, bool silent=false);
        #
        # In the case of the RooAbsArg overload, we need to tell Python that it
        # doesn't own the returned pointer. That's because the function name
        # contains "Clone", which makes cppyy guess that the returned pointer
        # points to a clone owned by the caller. In the case of the
        # RooAbsCollection input, the return value will be `None` and we don't
        # need to change any ownership flags (in fact, calling
        # SetOwnership(None, False) would cause a crash).
        if clonedArg is not None:
            ROOT.SetOwnership(clonedArg, False)

    def addOwned(self, arg, silent=False):

        import ROOT

        self._addOwned(arg, silent)
        ROOT.SetOwnership(arg, False)

    @cpp_signature(
        "RooAbsCollection::printLatex(const RooCmdArg& arg1={}, const RooCmdArg& arg2={},"
        "                        const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
        "                        const RooCmdArg& arg5={}, const RooCmdArg& arg6={},"
        "                        const RooCmdArg& arg7={}, const RooCmdArg& arg8={}) const ;"
    )
    def printLatex(self, *args, **kwargs):
        r"""The RooAbsCollection::printLatex() function is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the function.
        """
        # Redefinition of `RooAbsCollection.printLatex` for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        return self._printLatex(*args, **kwargs)
