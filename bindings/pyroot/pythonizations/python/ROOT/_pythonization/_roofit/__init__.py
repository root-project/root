# Authors:
# * Harshal Shende  03/2021
# * Hinnerk C. Schmidt 02/2021
# * Jonas Rembser 03/2021

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from .. import pythonization
from ._rooabscollection import RooAbsCollection
from ._rooabsdata import RooAbsData
from ._rooabspdf import RooAbsPdf
from ._rooabsreal import RooAbsReal
from ._rooabsreallvalue import RooAbsRealLValue
from ._rooarglist import RooArgList
from ._rooargset import RooArgSet
from ._roocategory import RooCategory
from ._roodatahist import RooDataHist
from ._roodataset import RooDataSet
from ._roodecays import RooBCPEffDecay, RooBCPGenDecay, RooBDecay, RooBMixDecay, RooDecay
from ._roogenfitstudy import RooGenFitStudy
from ._rooglobalfunc import (
    DataError,
    FitOptions,
    Format,
    Frame,
    Import,
    Link,
    MultiArg,
    Slice,
    YVar,
    ZVar,
    bindFunction,
)
from ._roojsonfactorywstool import RooJSONFactoryWSTool
from ._roomcstudy import RooMCStudy
from ._roomsgservice import RooMsgService
from ._rooprodpdf import RooProdPdf
from ._roorealvar import RooRealVar
from ._roosimultaneous import RooSimultaneous
from ._roosimwstool import RooSimWSTool
from ._roostats import SPlot
from ._roovectordatastore import RooVectorDataStore
from ._rooworkspace import RooWorkspace

# list of python classes that are used to pythonize RooFit classes
python_classes = [
    RooAbsCollection,
    RooAbsData,
    RooAbsPdf,
    RooAbsReal,
    RooAbsRealLValue,
    RooArgList,
    RooArgSet,
    RooBCPGenDecay,
    RooBCPEffDecay,
    RooBDecay,
    RooBMixDecay,
    RooCategory,
    RooDataHist,
    RooDataSet,
    RooDecay,
    RooGenFitStudy,
    RooJSONFactoryWSTool,
    RooMCStudy,
    RooMsgService,
    RooProdPdf,
    RooRealVar,
    RooSimultaneous,
    RooSimWSTool,
    RooWorkspace,
    RooVectorDataStore,
    SPlot,
]

# list of python functions that are used to pythonize RooGlobalFunc function in RooFit
python_roofit_functions = [
    DataError,
    FitOptions,
    Format,
    Frame,
    MultiArg,
    YVar,
    ZVar,
    Slice,
    Import,
    Link,
    bindFunction,
]

# create a dictionary for convenient access to python classes
python_classes_dict = dict()
for python_class in python_classes:
    python_classes_dict[python_class.__name__] = python_class


def get_defined_attributes(klass, consider_base_classes=False):
    """
    Get all class attributes that are defined in a given class or optionally in
    any of its base classes (except for `object`).
    """

    blacklist = [
        "__dict__",
        "__doc__",
        "__hash__",
        "__module__",
        "__weakref__",
        "__firstlineno__",
        "__static_attributes__",
        "__cpp_name__",
    ]

    if not consider_base_classes:
        return sorted([attr for attr in klass.__dict__.keys() if attr not in blacklist])

    # get a list of this class and all its base classes, excluding `object`
    method_resolution_order = klass.mro()
    if object in method_resolution_order:
        method_resolution_order.remove(object)

    def is_defined(funcname):

        if funcname in blacklist:
            return False

        in_any_dict = False

        for mro_class in method_resolution_order:
            if funcname in mro_class.__dict__:
                in_any_dict = True

        return in_any_dict

    return sorted([attr for attr in dir(klass) if is_defined(attr)])


def is_classmethod(klass, func):
    if hasattr(func, "__self__"):
        return func.__self__ == klass
    return False


def rebind_attribute(to_class, from_class, func_name):
    """
    Bind the instance method `from_class.func_name` also to class `to_class`.
    """
    from_method = getattr(from_class, func_name)

    if is_classmethod(from_class, from_method):
        # the @classmethod case
        to_method = classmethod(from_method.__func__)
    else:
        # any other case in Python 3 is trivial
        to_method = from_method

    setattr(to_class, func_name, to_method)


def make_func_name_orig(func_name):
    """Return the name that we will give to the original cppyy function."""
    # special treatment of magic functions, e.g.: __getitem__ > _getitem
    if func_name.startswith("__") and func_name.endswith("__"):
        func_name = func_name[2:-2]

    return "_" + func_name


@pythonization("Roo", is_prefix=True)
def pythonize_roofit_class(klass, name):
    # Parameters:
    # klass: class to pythonize
    # name: string containing the name of the class

    if name not in python_classes_dict:
        return

    python_klass = python_classes_dict[name]

    # list of functions to pythonize, which are assumed to be all functions in
    # that are manually defined in the Python classes or their superclasses
    func_names = get_defined_attributes(python_klass)

    for func_name in func_names:

        # if the RooFit class already has a function with the same name as our
        # pythonization, we rename it and prefix it with an underscore
        if hasattr(klass, func_name):
            # new name for original function
            func_name_orig = make_func_name_orig(func_name)
            func_orig = getattr(klass, func_name)
            func_new = getattr(python_klass, func_name)

            import inspect

            if func_new.__doc__ is None:
                func_new.__doc__ = func_orig.__doc__
            elif func_orig.__doc__ is not None:
                python_docstring = func_new.__doc__
                func_new.__doc__ = "Pythonization info\n"
                func_new.__doc__ += "==============\n\n"
                func_new.__doc__ += inspect.cleandoc(python_docstring) + "\n\n"
                func_new.__doc__ += "Documentation of original cppyy.CPPOverload object\n"
                func_new.__doc__ += "==================================================\n\n"
                func_new.__doc__ += func_orig.__doc__

            setattr(klass, func_name_orig, func_orig)

        rebind_attribute(klass, python_klass, func_name)

    return


def pythonize_roofit_namespace(ns):

    for python_func in python_roofit_functions:
        func_name = python_func.__name__
        if hasattr(ns, func_name):
            func_name_orig = "_" + func_name
            setattr(ns, func_name_orig, getattr(ns, func_name))
        setattr(ns, func_name, python_func)

    return ns
