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


r"""
/**
\class RooGlobalFunc
\brief \parblock \endparblock
\htmlonly
<div class="pyrootbox">
\endhtmlonly

## PyROOT

Some member functions of RooGlobalFunc that take a RooCmdArg as argument also support keyword arguments.
So far, this applies to FitOptions, Format, Frame, MultiArg, YVar and ZVar.
\code{.py}
# Directly passing a RooCmdArg:
ROOT.RooMCStudy(model, ROOT.RooArgSet(x), ROOT.RooFit.FitOptions=(ROOT.RooFit.Save(True), ROOT.RooFit.PrintEvalErrors(0)))

# With keyword arguments:
ROOT.RooMCStudy(model, ROOT.RooArgSet(x), FitOptions=dict(Save=True, PrintEvalErrors=0))
\endcode

\htmlonly
</div>
\endhtmlonly
*/
"""

from ._utils import _kwargs_to_roocmdargs, _dict_to_flat_map, cpp_signature


@cpp_signature(
    "RooFit::FitOptions(const RooCmdArg& arg1, const RooCmdArg& arg2={},"
    "const RooCmdArg& arg3={},const RooCmdArg& arg4={},"
    "const RooCmdArg& arg5={},const RooCmdArg& arg6={}) ;"
)
def FitOptions(*args, **kwargs):
    r"""The FitOptions() function is pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the function.
    """
    # Redefinition of `FitOptions` for keyword arguments.
    from cppyy.gbl import RooFit

    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return RooFit._FitOptions(*args, **kwargs)


@cpp_signature(
    "RooFit::Format(const char* what, const RooCmdArg& arg1={}, const RooCmdArg& arg2={},"
    "const RooCmdArg& arg3={},const RooCmdArg& arg4={},"
    "const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
    "const RooCmdArg& arg7={},const RooCmdArg& arg8={}) ;"
)
def Format(*args, **kwargs):
    r"""The Format() function is pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the function.
    """
    # Redefinition of `Format` for keyword arguments.
    from cppyy.gbl import RooFit

    if "what" in kwargs:
        args = (kwargs["what"],) + args
        del kwargs["what"]
    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return RooFit._Format(*args, **kwargs)


@cpp_signature(
    "RooFit::Frame(const RooCmdArg& arg1, const RooCmdArg& arg2={},"
    "const RooCmdArg& arg3={}, const RooCmdArg& arg4={},"
    "const RooCmdArg& arg5={}, const RooCmdArg& arg6={}) ;"
)
def Frame(*args, **kwargs):
    r"""The Frame() function is pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the function.
    """
    # Redefinition of `Frame` for keyword arguments.
    from cppyy.gbl import RooFit

    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return RooFit._Frame(*args, **kwargs)


@cpp_signature(
    "RooFit::MultiArg(const RooCmdArg& arg1, const RooCmdArg& arg2,"
    "const RooCmdArg& arg3={},const RooCmdArg& arg4={},"
    "const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
    "const RooCmdArg& arg7={},const RooCmdArg& arg8={}) ;"
)
def MultiArg(*args, **kwargs):
    r"""The MultiArg() function is pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the function.
    """
    # Redefinition of `MultiArg` for keyword arguments.
    from cppyy.gbl import RooFit

    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return RooFit._MultiArg(*args, **kwargs)


@cpp_signature("RooFit::YVar(const RooAbsRealLValue& var, const RooCmdArg& arg={}) ;")
def YVar(*args, **kwargs):
    r"""The YVar() function is pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the function.
    """
    # Redefinition of `YVar` for keyword arguments.
    from cppyy.gbl import RooFit

    if "var" in kwargs:
        args = (kwargs["var"],) + args
        del kwargs["var"]
    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return RooFit._YVar(*args, **kwargs)


@cpp_signature("RooFit::ZVar(const RooAbsRealLValue& var, const RooCmdArg& arg={}) ;")
def ZVar(*args, **kwargs):
    r"""The ZVar() function is pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the function.
    """
    # Redefinition of `ZVar` for keyword arguments.
    from cppyy.gbl import RooFit

    if "var" in kwargs:
        args = (kwargs["var"],) + args
        del kwargs["var"]
    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return RooFit._ZVar(*args, **kwargs)


@cpp_signature("RooFit::Slice(std::map<RooCategory*, std::string> const&) ;")
def Slice(*args, **kwargs):
    r"""The Slice function is pythonized for converting python dict to std::map.
    The keywords must correspond to the CmdArg of the function.
    The instances in the dict must correspond to the template argument in std::map of the function.
    """
    # Redefinition of `Slice` for keyword arguments and converting python dict to std::map.
    from cppyy.gbl import RooFit

    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
        return RooFit.Detail.SliceFlatMap(_dict_to_flat_map(args[0], {"RooCategory*": "std::string"}))

    return RooFit._Slice(*args, **kwargs)


@cpp_signature(
    [
        "RooFit::Import(const std::map<std::string,RooDataSet*>& ) ;",
        "RooFit::Import(const std::map<std::string,TH1*>&) ;",
        "RooFit::Import(const std::map<std::string,RooDataHist*>&) ;",
    ]
)
def Import(*args, **kwargs):
    r"""The Import function is pythonized for converting python dict to std::map.
    The keywords must correspond to the CmdArg of the function.
    The instances in the dict must correspond to the template argument in std::map of the function.
    """
    # Redefinition of `Import` for keyword arguments and converting python dict to std::map.
    from cppyy.gbl import RooFit

    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
        return RooFit.Detail.ImportFlatMap(
            _dict_to_flat_map(args[0], {"std::string": ["TH1*", "RooDataHist*", "RooDataSet*"]})
        )

    return RooFit._Import(*args, **kwargs)


@cpp_signature("RooFit::Link(const std::map<std::string,RooAbsData*>&) ;")
def Link(*args, **kwargs):
    r"""The Link function is pythonized for converting python dict to std::map.
    The keywords must correspond to the CmdArg of the function.
    The instances in the dict must correspond to the template argument in std::map of the function.
    """
    # Redefinition of `Link` for keyword arguments and converting python dict to std::map.
    from cppyy.gbl import RooFit

    if len(args) == 1 and len(kwargs) == 0 and isinstance(args[0], dict):
        return RooFit.Detail.LinkFlatMap(_dict_to_flat_map(args[0], {"std::string": "RooAbsData*"}))

    return RooFit._Link(*args, **kwargs)


@cpp_signature("RooFit::DataError(Int_t) ;")
def DataError(etype):
    r"""If you want to use the `"None"` enum value to disable error plotting, you
    can also pass `None` directly instead of passing a string:

    ~~~ {.py}
        data.plotOn(frame, DataError=None)
        # instead of DataError="None"
    ~~~
    """
    from cppyy.gbl import RooFit

    # One of the possible enum values is "None", and we want the user to be
    # able to pass None also as a NoneType for convenience.
    if etype is None:
        etype = "None"

    return RooFit._DataError(etype)


def _bindFunctionOrPdf(name, func, is_rooabspdf, *variables):
    """
    Wrap an arbitrary function defined in Python or C++.

    If you're wrapping a Python function, it must take numpy arrays of type
    float64 as input and output types.

    Parameters:
    - name (str): Name of the function.
    - func (callable): Function that defines the function.
    - variables (list): List of variables to be used in the function.

    Returns:
    - RooAbsReal wrapping the given function
    """

    import ROOT
    import numpy as np

    # use the C++ version if dealing with C++ function
    if "cppyy" in repr(type(func)):
        return ROOT.RooFit._bindFunction(name, func, *variables)

    base_class_name = "RooAbsPdf" if is_rooabspdf else "RooAbsReal"

    class RooPyBindDerived(ROOT.RooFit.Detail.RooPyBind[base_class_name]):

        _outputBuffer = None

        def __init__(self, name, title, *variables):
            super(RooPyBindDerived, self).__init__(name, title, ROOT.RooArgList(*variables))

        def evaluate(self):
            inputs = [np.array([v.getVal()]) for v in self.varlist()]
            return func(*inputs)[0]

        def doEvalPy(self, ctx):

            def span_to_numpy(sp):
                return np.frombuffer(sp.data(), dtype=np.float64, count=sp.size())

            inputs = [span_to_numpy(ctx.at(v)) for v in self.varlist()]
            if self._outputBuffer is None:
                self._outputBuffer = np.zeros(ctx.output().size())
            self._outputBuffer[:] = func(*inputs)
            return self._outputBuffer

        def clone(self, newname=False):
            cl = RooPyBindDerived(newname if newname else self.GetName(), self.GetTitle(), self.varlist())
            ROOT.SetOwnership(cl, False)
            return cl

    return RooPyBindDerived(name, "", variables)


def bindFunction(name, func, *variables):
    return _bindFunctionOrPdf(name, func, False, *variables)


def bindPdf(name, func, *variables):
    return _bindFunctionOrPdf(name, func, True, *variables)
