# Author: Enric Tejedor, Danilo Piparo CERN  06/2018

################################################################################
# Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################
import importlib
import inspect
import pkgutil
import traceback

import cppyy

from ._generic import pythonize_generic

# \cond INTERNALS
gbl_namespace = cppyy.gbl
# \endcond


def pythonization(class_name, ns="::", is_prefix=False):
    r"""
    \ingroup Pythonizations
    Decorator that allows to pythonize C++ classes. To pythonize means to add
    some extra behaviour to a C++ class that is used from Python via PyROOT,
    so that such a class can be used in an easier / more "pythonic" way.
    When a pythonization is registered with this decorator, the injection of
    the new behaviour in the C++ class is done immediately, if the class has
    already been used from the application, or lazily, i.e. only when the class
    is first accessed from the application.

    Args:
        class_name (string/iterable[string]): specifies either a single string or
            multiple strings, where each string can be either (i) the name of a
            C++ class to be pythonized, or (ii) a prefix to match all classes
            whose name starts with that prefix.
        ns (string): namespace of the classes to be pythonized. Default is the
            global namespace (`::`).
        is_prefix (boolean): if True, `class_name` contains one or multiple
            prefixes, each prefix potentially matching multiple classes.
            Default is False.
            These are examples of prefixes and namespace and what they match:
            - class_name="", ns="::" : all classes in the global namespace.
            - class_name="C", ns="::" : all classes in the global namespace
              whose name starts with "C"
            - class_name="", ns="NS1::NS2" : all classes in namespace "NS1::NS2"
            - class_name="C", ns="NS1::NS2" : all classes in namespace
              "NS1::NS2" whose name starts with "C"

    Returns:
        function: function that receives the user-defined function and
            decorates it.

    """

    # Type check and parsing of target argument.
    # Retrieve the scope(s) of the class(es)/prefix(es) to register the
    # pythonizor in the right scope(s)
    target = _check_target(class_name)

    # Remove trailing '::' from namespace
    if ns.endswith("::"):
        ns = ns[:-2]

    # Create a filter lambda for the target class(es)/prefix(es)
    if is_prefix:

        def passes_filter(class_name):
            return any(class_name.startswith(prefix) for prefix in target)
    else:

        def passes_filter(class_name):
            return class_name in target

    def pythonization_impl(user_pythonizor):
        """
        The real decorator. Accepts a user-provided function and decorates it.
        An inner function - a wrapper of the user function - is registered in
        cppyy as a pythonizor.

        Args:
            user_pythonizor (function): user-provided function to be decorated.
                It implements some pythonization. It can accept two parameters:
                the class to be pythonized, i.e. the Python proxy of the class
                in which new behaviour can be injected, and optionally the name
                of that class (can be used e.g. to do some more complex
                filtering).

        Returns:
            function: the user function, after being registered as a
                pythonizor.
        """

        npars = _check_num_pars(user_pythonizor)

        # Check whether any of the target classes has already been used.
        # If so, the class proxy has to be immediately pythonized - even if we
        # registered a pythonizor for it, the pythonizor would never be executed
        _find_used_classes(ns, passes_filter, user_pythonizor, npars)

        def cppyy_pythonizor(klass, name):
            """
            Wrapper function with the parameters that cppyy requires for a
            pythonizor function (class proxy and class name). It invokes the
            user function only if the current class - a candidate for being
            pythonized - matches the `target` argument of the decorator.

            Args:
                klass (class type): cppyy proxy of the class that is the
                    current candidate to be pythonized.
                name (string): name of the class that is the current candidate
                    to be pythonized.
            """

            fqn = klass.__cpp_name__

            # Add pretty printing (done on all classes)
            pythonize_generic(klass, fqn)

            if passes_filter(name):
                _invoke(user_pythonizor, npars, klass, fqn)

        # Register pythonizor in its namespace
        cppyy.py.add_pythonization(cppyy_pythonizor, ns)

        # Return the original user function.
        # We don't want to modify the user function, we just use the decorator
        # to register the function as a pythonizor.
        # This allows for correct chaining of multiple @pythonization decorators
        # for a single function
        return user_pythonizor

    return pythonization_impl


# \cond INTERNALS


def _check_target(target):
    """
    Helper function to check the type of the `class name` argument specified by
    the user in a @pythonization decorator.

    Args:
        target (string/iterable[string]): class name(s)/prefix(es).

    Returns:
        list[string]: class name(s)/prefix(es) in `target`, with no repetitions.
    """

    if isinstance(target, str):
        _check_no_namespace(target)
        target = [target]
    else:
        for name in target:
            if isinstance(name, str):
                _check_no_namespace(name)
            else:
                raise TypeError(
                    'Invalid type of "target" argument in @pythonization: must be string or iterable of strings'
                )
        # Remove possible duplicates
        target = list(set(target))

    return target


def _check_no_namespace(target):
    """
    Checks that a given target of a pythonizor does not specify a namespace
    (only the class name / prefix of a class name should be present).

    Args:
        target (string): class name/prefix.
    """

    if target.find("::") >= 0:
        raise ValueError(
            'Invalid value of "class_name" argument in '
            '@pythonization: namespace definition found ("{}"). '
            'Please use the "ns" parameter to specify the '
            "namespace".format(target)
        )


def _check_num_pars(f):
    """
    Checks the number of parameters of the `f` function.

    Args:
        f (function): user pythonizor function.

    Returns:
        int: number of positional parameters of `f`.
    """
    npars = len(inspect.getfullargspec(f).args)
    if npars == 0 or npars > 2:
        raise TypeError(
            "Pythonizor function {} has a wrong number of "
            "parameters ({}). Allowed parameters are the class to "
            "be pythonized and (optionally) its name.".format(f.__name__, npars)
        )

    return npars


def _invoke(user_pythonizor, npars, klass, fqn):
    """
    Invokes the given user pythonizor function with the right arguments.

    Args:
        user_pythonizor (function): user pythonizor function.
        npars (int): number of parameters of the user pythonizor function.
        klass (class type): cppyy proxy of the class to be pythonized.
        fqn (string): fully-qualified name of the class to be pythonized.
    """

    try:
        if npars == 1:
            user_pythonizor(klass)
        else:
            user_pythonizor(klass, fqn)
    except Exception:
        print("Error pythonizing class {}:".format(fqn))
        traceback.print_exc()
        # Propagate the error so that the class lookup that triggered this
        # pythonization fails too and the application stops
        raise RuntimeError


def _find_used_classes(ns, passes_filter, user_pythonizor, npars):
    """
    Finds already instantiated classes in namespace `ns` that pass the filter
    of `passes_filter`. Every matching class is pythonized with the
    `user_pythonizor` function.
    This makes sure a pythonizor is also applied to classes that have already
    been used at the time the pythonizor is registered.

    Args:
        ns (string): namespace of the class names of prefixes in `targets`.
        passes_filter (function): function that determines if a given class
            is the target of `user_pythonizor`.
        user_pythonizor (function): user pythonizor function.
        npars (int): number of parameters of the user pythonizor function.
    """

    ns_obj = _find_namespace(ns)
    if ns_obj is None:
        # Namespace has not been used yet, no need to inspect more
        return

    def pythonize_if_match(name, klass):
        # Check if name matches, excluding the namespace
        if passes_filter(name.split("::")[-1]):
            # Pythonize right away!
            _invoke(user_pythonizor, npars, klass, klass.__cpp_name__)

    def get_class_name(instantiation):
        # Get the right class name for the input instantiation

        # Template instantiation such as cppyy.gbl.MyClass["SomeType"]
        if isinstance(instantiation, str):
            return instantiation

        # Template instantiation such as cppyy.gbl.MyClass[cppyy.gbl.SomeType]
        # use the more specialized attribute first, then a more generic one
        if hasattr(instantiation, "__cpp_name__"):
            return instantiation.__cpp_name__

        if hasattr(instantiation, "__name__"):
            return instantiation.__name__

        raise RuntimeError(
            f"The template instantiation '{instantiation}' cannot be properly pythonized. Please report this as a bug."
        )

    ns_vars = vars(ns_obj)
    for var_name, var_value in ns_vars.items():
        if str(var_value).startswith("<class cppyy.gbl."):
            # It's a class proxy
            pythonize_if_match(var_name, var_value)

        if str(var_value).startswith("<cppyy.Template"):
            # If this is a template, pythonize the instances. Note that in
            # older cppyy, template instantiations are cached by
            # fully-qualified name directly in the namespace, so they are
            # covered by the code branch above.
            instantiations = getattr(var_value, "_instantiations", {})
            for args, instance in instantiations.items():
                # Make sure we don't do any redundant pythonization, e.g. if we
                # use a version of cppyy that caches both in the namespace and
                # in the _instantiations attribute.
                if instance not in ns_vars:
                    instance_name = var_name + "<" + ",".join(map(get_class_name, args)) + ">"
                    pythonize_if_match(instance_name, instance)


def _find_namespace(ns):
    """
    Finds and returns the proxy object of the `ns` namespace, if it has already
    been accessed.

    Args:
        ns (string): a namespace.

    Returns:
        namespace proxy object, if the namespace has already been accessed,
            otherwise None.
    """

    if ns == "":
        return gbl_namespace

    ns_obj = gbl_namespace
    # Get all namespaces in a list
    every_ns = ns.split("::")
    for ns in every_ns:
        ns_vars = vars(ns_obj)
        if ns not in ns_vars:
            return None
        ns_obj = getattr(ns_obj, ns)

    return ns_obj


def _register_pythonizations():
    """
    Registers the ROOT pythonizations with cppyy for lazy injection.
    """

    exclude = ["_rdf_utils", "_rdf_pyz", "_rdf_conversion_maps"]
    for _, module_name, _ in pkgutil.walk_packages(__path__):
        if module_name not in exclude:
            importlib.import_module(__name__ + "." + module_name)


# \endcond
