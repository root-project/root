# Authors:
# * Harshal Shende  04/2022
# * Lorenzo Moneta 04/2022

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

import sys
import cppyy
from cppyy.gbl import gSystem

from .. import pythonization

from ._factory import Factory
from ._dataloader import DataLoader
from ._crossvalidation import CrossValidation

from ._rbdt import Compute, pythonize_rbdt


def inject_rbatchgenerator(ns):
    from ._batchgenerator import (
        CreateNumPyGenerators,
        CreateTFDatasets,
        CreatePyTorchGenerators,
    )

    python_batchgenerator_functions = [
        CreateNumPyGenerators,
        CreateTFDatasets,
        CreatePyTorchGenerators,
    ]

    for python_func in python_batchgenerator_functions:
        func_name = python_func.__name__
        setattr(ns.Experimental, func_name, python_func)

    return ns


from ._gnn import RModel_GNN, RModel_GraphIndependent

hasRDF = "dataframe" in cppyy.gbl.ROOT.GetROOT().GetConfigFeatures()
if hasRDF:
    from ._rtensor import (
        get_array_interface,
        add_array_interface_property,
        RTensorGetitem,
        pythonize_rtensor,
        _AsRTensor,
    )

# this should be available only when xgboost is there ?
# We probably don't need a protection here since the code is run only when there is xgboost
from ._tree_inference import SaveXGBoost


# list of python classes that are used to pythonize TMVA classes
python_classes = [Factory, DataLoader, CrossValidation]

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

    import sys

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


@pythonization(class_name=["Factory", "DataLoader", "CrossValidation"], ns="TMVA")
def pythonize_tmva(klass, name):
    # Parameters:
    # klass: class to pythonize
    # name: string containing the name of the class

    # need to strip the TMVA namespace
    ns_prefix = "TMVA::"
    name = name[len(ns_prefix) : len(name)]

    if not name in python_classes_dict:
        print("Error - class ", name, "is not in the pythonization list")
        return

    python_klass = python_classes_dict[name]

    # list of functions to pythonize, which are assumed to be all functions in
    # that are manually defined in the Python classes or their superclasses
    func_names = get_defined_attributes(python_klass)

    for func_name in func_names:

        # if the TMVA class already has a function with the same name as our
        # pythonization, we rename it and prefix it with an underscore
        if hasattr(klass, func_name):
            # new name for original function
            func_name_orig = make_func_name_orig(func_name)
            func_orig = getattr(klass, func_name)
            func_new = getattr(python_klass, func_name)

            import inspect

            if func_new.__doc__ is None:
                func_new.__doc__ = func_orig.__doc__
            elif not func_orig.__doc__ is None:
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
