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

from ._utils import _kwargs_to_roocmdargs, _string_to_root_attribute, _dict_to_std_map, cpp_signature


# Color and Style dictionary to define matplotlib conventions
_color_map = {
    "r": "kRed",
    "b": "kBlue",
    "g": "kGreen",
    "y": "kYellow",
    "w": "kWhite",
    "k": "kBlack",
    "m": "kMagenta",
    "c": "kCyan",
}
_style_map = {"-": "kSolid", "--": "kDashed", ":": "kDotted", "-.": "kDashDotted"}


@cpp_signature(
    "RooFit::FitOptions(const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),"
    "const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),"
    "const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none()) ;"
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
    "RooFit::Format(const char* what, const RooCmdArg& arg1=RooCmdArg::none(), const RooCmdArg& arg2=RooCmdArg::none(),"
    "const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),"
    "const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none(),"
    "const RooCmdArg& arg7=RooCmdArg::none(),const RooCmdArg& arg8=RooCmdArg::none()) ;"
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
    "RooFit::Frame(const RooCmdArg& arg1, const RooCmdArg& arg2=RooCmdArg::none(),"
    "const RooCmdArg& arg3=RooCmdArg::none(), const RooCmdArg& arg4=RooCmdArg::none(),"
    "const RooCmdArg& arg5=RooCmdArg::none(), const RooCmdArg& arg6=RooCmdArg::none()) ;"
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
    "const RooCmdArg& arg3=RooCmdArg::none(),const RooCmdArg& arg4=RooCmdArg::none(),"
    "const RooCmdArg& arg5=RooCmdArg::none(),const RooCmdArg& arg6=RooCmdArg::none(),"
    "const RooCmdArg& arg7=RooCmdArg::none(),const RooCmdArg& arg8=RooCmdArg::none()) ;"
)
def MultiArg(*args, **kwargs):
    r"""The MultiArg() function is pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the function.
    """
    # Redefinition of `MultiArg` for keyword arguments.
    from cppyy.gbl import RooFit

    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return RooFit._MultiArg(*args, **kwargs)


@cpp_signature("RooFit::YVar(const RooAbsRealLValue& var, const RooCmdArg& arg=RooCmdArg::none()) ;")
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


@cpp_signature("RooFit::ZVar(const RooAbsRealLValue& var, const RooCmdArg& arg=RooCmdArg::none()) ;")
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
        args = list(args)
        args[0] = _dict_to_std_map(args[0], {"RooCategory*": "std::string"})
        return RooFit._Slice(args[0])

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
        args = list(args)
        args[0] = _dict_to_std_map(args[0], {"std::string": ["TH1*", "RooDataHist*", "RooDataSet*"]})
        return RooFit._Import(args[0])

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
        args = list(args)
        args[0] = _dict_to_std_map(args[0], {"std::string": "RooAbsData*"})
        return RooFit._Link(args[0])

    return RooFit._Link(*args, **kwargs)


@cpp_signature("RooFit::LineColor(Color_t color) ;")
def LineColor(color):
    r"""The `color` argument doesn't necessarily have to be a ROOT color enum value, like `ROOT.kRed`.
    Here is what you can also do in PyROOT:

      1. Pass a string with the enum value name instead, e.g.:
    ~~~ {.py}
    pdf.plotOn(frame, LineColor="kRed")
    ~~~
      2. Pass a string with the corresponding single-character color code following the matplotlib convention:
    ~~~ {.py}
    pdf.plotOn(frame, LineColor="r")
    ~~~
      3. Pass a string with the enum value name instead followed by some manipulation of the enum value:
    ~~~ {.py}
    pdf.plotOn(frame, LineColor="kRed+1")
    ~~~
    """
    from cppyy.gbl import RooFit

    return RooFit._LineColor(_string_to_root_attribute(color, _color_map))


@cpp_signature("RooFit::FillColor(Color_t color) ;")
def FillColor(color):
    # Redefinition of `FillColor` for matplotlib conventions and string arguments.
    from cppyy.gbl import RooFit

    return RooFit._FillColor(_string_to_root_attribute(color, _color_map))

# Copy the docstring from LineColor.
FillColor.__doc__ = LineColor.__doc__


@cpp_signature("RooFit::MarkerColor(Color_t color) ;")
def MarkerColor(color):
    # Redefinition of `MarkerColor` for matplotlib conventions and string arguments.
    from cppyy.gbl import RooFit

    return RooFit._MarkerColor(_string_to_root_attribute(color, _color_map))

# Copy the docstring from LineColor.
MarkerColor.__doc__ = LineColor.__doc__


@cpp_signature("RooFit::LineStyle(Style_t style) ;")
def LineStyle(style):
    # Redefinition of `LineStyle` for matplotlib conventions and string arguments.
    from cppyy.gbl import RooFit

    return RooFit._LineStyle(_string_to_root_attribute(style, _style_map))


@cpp_signature("RooFit::FillStyle(Style_t style) ;")
def FillStyle(style):
    # Redefinition of `FillStyle` for matplotlib conventions and string arguments.
    from cppyy.gbl import RooFit

    return RooFit._FillStyle(_string_to_root_attribute(style, {}))


@cpp_signature("RooFit::MarkerStyle(Style_t style) ;")
def MarkerStyle(style):
    # Redefinition of `MarkerStyle` for matplotlib conventions and string arguments.
    from cppyy.gbl import RooFit

    return RooFit._MarkerStyle(_string_to_root_attribute(style, {}))


@cpp_signature("RooFit::DataError(Int_t) ;")
def DataError(etype):
    r"""Instead of passing an enum value to this function, you can pass a
    string with the name of that enum value, for example:

    ~~~ {.py}
        data.plotOn(frame, DataError="SumW2")
        # instead of DataError=ROOT.RooAbsData.SumW2
    ~~~

    If you want to use the `"None"` enum value to disable error plotting, you
    can also pass `None` directly instead of passing a string:

    ~~~ {.py}
        data.plotOn(frame, DataError=None)
        # instead of DataError="None"
    ~~~
    """
    # Redefinition of `DataError` to also accept `str` or `NoneType` to get the
    # corresponding enum values from RooAbsData.DataError.
    from cppyy.gbl import RooFit

    # One of the possible enum values is "None", and we want the user to be
    # able to pass None also as a NoneType for convenience.
    if etype is None:
        etype = "None"

    if isinstance(etype, str):
        try:
            import ROOT
            etype = getattr(ROOT.RooAbsData.ErrorType, etype)
        except AttributeError as error:
            raise ValueError(
                'Unsupported error type type passed to DataError().'
                + ' Supported decay types are : "Poisson", "SumW2", "Auto", "Expected", and None.'
            )
        except Exception as exception:
            raise exception

    return RooFit._DataError(etype)
