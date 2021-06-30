# Authors:
# * Hinnerk C. Schmidt 02/2021
# * Jonas Rembser 06/2021
# * Harshal Shende 06/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


def _getter(k, v):
    # helper function to get CmdArg attribute from `RooFit`
    # Parameters:
    # k: key of the kwarg
    # v: value of the kwarg

    # We have to use ROOT here and not cppy.gbl, because the RooFit namespace is pythonized itself.
    import ROOT
    import libcppyy

    func = getattr(ROOT.RooFit, k)

    if isinstance(func, libcppyy.CPPOverload):
        # Pythonization for functions that don't pass any RooCmdArgs like ShiftToZero() and MoveToBack(). For Eg,
        # Default bindings: pdf.plotOn(frame, ROOT.RooFit.MoveToBack())
        # With pythonizations: pdf.plotOn(frame, MoveToBack=True)

        if "()" in func.func_doc:
            if not isinstance(v, bool):
                raise TypeError("The keyword argument " + k + " can only take bool values.")
            return func() if v else ROOT.RooCmdArg.none()

    if isinstance(v, (tuple, list)):
        return func(*v)
    elif isinstance(v, (dict,)):
        return func(**v)
    else:
        return func(v)


def _kwargs_to_roocmdargs(*args, **kwargs):
    """Helper function to check kwargs and pythonize the arguments using _getter"""
    if kwargs:
        args = args + tuple((_getter(k, v) for k, v in kwargs.items()))
    return args, {}


def _string_to_root_attribute(value, lookup_map):
    """Helper function to pythonize arguments based on the matplotlib color/style conventions."""
    # lookup_map for color and style defined in _rooglobalfunc.py have matplotlib conventions specified which enables
    # the use of string values like "r" or ":" instead of using enums like ROOT.kRed or ROOT.kDotted for colors or styles. For Eg.
    # Default bindings:  pdf.plotOn(frame, LineColor=ROOT.kOrange)
    # With pythonizations: pdf.plotOn(frame, LineColor="kOrange")

    import ROOT

    if isinstance(value, str):
        if value in lookup_map:
            return getattr(ROOT, lookup_map[value])
        else:
            try:
                return getattr(ROOT, value)
            except:
                raise ValueError(
                    "Unsupported value passed. The value either has to be the name of an attribute of the ROOT module, or match with one of the following values that get translated to ROOT attributes: {}".format(
                        _lookup_map
                    )
                )
    else:
        return value


def _decaytype_string_to_enum(caller, kwargs):
    """Helper function to pythonize DecayType enums and check for enum value names."""
    type_key = "type"

    if type_key in kwargs:
        val = kwargs[type_key]
        if isinstance(val, str):
            try:
                kwargs[type_key] = getattr(caller.__class__, val)
            except AttributeError as error:
                raise ValueError(
                    "Unsupported decay type passed to "
                    + caller.__class__.__name__
                    + ". Supported decay types are : 'SingleSided', 'DoubleSided', 'Flipped'"
                )
            except Exception as exception:
                raise exception

    return kwargs
