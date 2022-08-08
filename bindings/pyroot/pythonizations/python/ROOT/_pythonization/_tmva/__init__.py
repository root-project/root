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

from .. import pythonization

from ._factory import Factory
from ._dataloader import DataLoader
from ._crossvalidation import CrossValidation
from ._rbdt import *
from ._rtensor import *
from ._tree_inference import *


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

    blacklist = ["__dict__", "__doc__", "__hash__", "__module__", "__weakref__"]

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


def is_staticmethod_py2(klass, func_name):
    """Check if the function with name `func_name` of a class is a static method in Python 2."""

    return type(getattr(klass, func_name)).__name__ == "function"


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
    elif sys.version_info >= (3, 0):
        # any other case in Python 3 is trivial
        to_method = from_method
    elif isinstance(from_method, property):
        # the @property case in Python 2
        to_method = from_method
    elif is_staticmethod_py2(from_class, func_name):
        # the @staticmethod case in Python 2
        to_method = staticmethod(from_method)
    else:
        # the instance method case in Python 2
        import new

        to_method = new.instancemethod(from_method.__func__, None, to_class)

    setattr(to_class, func_name, to_method)


def make_func_name_orig(func_name):
    """Return the name that we will give to the original cppyy function."""
    # special treatment of magic functions, e.g.: __getitem__ > _getitem
    if func_name.startswith("__") and func_name.endswith("__"):
        func_name = func_name[2:-2]

    return "_" + func_name


@pythonization(class_name=["Factory","DataLoader","CrossValidation"], ns="TMVA")
def pythonize_tmva(klass, name):
    # Parameters:
    # klass: class to pythonize
    # name: string containing the name of the class

    # need to strip the TMVA namespace
    ns_prefix = "TMVA::"
    name = name[len(ns_prefix):len(name)]

    if not name in python_classes_dict:
        print("Error - class ",name,"is not in the pythonization list")
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
            import sys

            if sys.version_info < (3, 0):
                func_new = func_new.__func__

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