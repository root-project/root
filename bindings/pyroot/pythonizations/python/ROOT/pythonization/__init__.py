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
import re
import sys

import cppyy

from ._generic import pythonize_generic


def pythonization(target, is_prefix=False):
    '''
    Decorator that allows to pythonize C++ classes. To pythonize means to add
    some extra behaviour to a C++ class that is used from Python via PyROOT,
    so that such a class can be used in an easier / more "pythonic" way.
    The injection of the pythonization behaviour in the C++ class is done
    lazily, i.e. only when the class is first accessed from the application.

    Args:
        target (string/iterable[string]): specifies either a single string or
            multiple strings, where each string can be either (i) the
            fully-qualified name of a C++ class to be pythonized, or (ii) a
            prefix to match all classes whose fully-qualified name starts with
            that prefix.

            These are examples of prefixes and what they match:
            - "" : all classes in the global namespace
            - "C" : all classes in the global namespace whose name starts with
              "C"
            - "NS1::NS2::" : all classes in namespace "NS1::NS2"
            - "NS1::NS2::C" : all classes in namespace "NS1::NS2" whose name
              starts with "C"

        is_prefix (boolean): if True, `target` contains one or multiple
            prefixes, each prefix potentially matching multiple classes.
            Default is False.

    Returns:
        function: function that receives the user-defined function and
            decorates it.
    '''

    # Type check and parsing of target argument.
    # Retrieve the scope(s) of the class(es)/prefix(es) to register the
    # pythonizor in the right scope(s)
    scopes, scope_to_targets = _parse_target(target)

    # Create a filter lambda for the target class(es)/prefix(es)
    if is_prefix:
        def passes_filter(class_name, scope):
            prefixes = scope_to_targets.get(scope)
            if prefixes is None:
                return False # no prefix registered for this scope
            else:
                return any(class_name.startswith(prefix)
                           for prefix in prefixes)
    else:
        def passes_filter(class_name, scope):
            classes = scope_to_targets.get(scope)
            if classes is None:
                return False # no class registered for this scope
            else:
                return class_name in classes

    def pythonization_impl(user_pythonizor):
        '''
        The real decorator. Accepts a user-provided function and decorates it.
        An inner function - a wrapper of the user function - is registered in
        cppyy as a pythonizor.

        Args:
            user_pythonizor (function): user-provided function to be decorated.
                It implements some pythonization. It must accept two
                parameters: the class to be pythonized, i.e. the Python proxy
                of the class in which new behaviour can be injected, and the
                name of that class (can be used e.g. to do some more complex
                filtering).
        '''

        npars = _check_num_pars(user_pythonizor)

        def cppyy_pythonizor(klass, name):
            '''
            Wrapper function with the parameters that cppyy requires for a
            pythonizor function (class proxy and class name). It invokes the
            user function only if the current class - a candidate for being
            pythonized - matches the `target` argument of the decorator.

            Args:
                klass (class type): cppyy proxy of the class that is the
                    current candidate to be pythonized.
                name (string): name of the class that is the current candidate
                    to be pythonized.
            '''

            fqn = klass.__cpp_name__

            # Add pretty printing (done on all classes)
            pythonize_generic(klass, fqn)

            if passes_filter(name, _get_scope(fqn)):
                if npars == 1:
                    user_pythonizor(klass)
                else:
                    user_pythonizor(klass, fqn)

        # Register pythonizor in all the scopes of the requested classes
        for scope in scopes:
            cppyy.py.add_pythonization(cppyy_pythonizor, scope)

        return cppyy_pythonizor

    return pythonization_impl

def _parse_target(target):
    '''
    Helper function to check the type of the `target` argument specified by the
    user in a @pythonization decorator.
    It also returns the received fully-qualified class name(s)/prefix(es) and
    the corresponding scope(s), removing any repetition.

    Args:
        target (string/iterable[string]): fully-qualified class name(s)/
            prefix(es).

    Returns:
        tuple[set,set]: fully-qualified class name(s)/prefix(es) and scope(s)
            in `target`, with no repetitions.
    '''

    scopes = set()
    scope_to_targets = {}

    if _is_string(target):
        _register_target(target, scopes, scope_to_targets)
    else:
        try:
            for name in target:
                if _is_string(name):
                    _register_target(name, scopes, scope_to_targets)
                else:
                    raise TypeError()
        except TypeError:
            raise TypeError('Invalid type of "target" argument in @pythonization: '
                            'must be string or iterable of strings')

    return scopes, scope_to_targets

def _is_string(o):
    '''
    Checks if object is a string.

    Args:
        o (any type): object to be checked.

    Returns:
        True if object is a string, False otherwise.
    '''

    if sys.version_info >= (3, 0):
        return isinstance(o, str)
    else:
        return isinstance(o, basestring)

def _register_target(target, scopes, scope_to_targets):
    '''
    Registers all the targets of a pythonizor per scope.

    Args:
        target (string): fully-qualified class name/prefix.
        scopes (list): list of scopes to add the scope of `target` to.
        scope_to_targets (dict): map from scope to list of targets that have
            been registered for that scope.
    '''

    scope, class_name = _split_scope_and_class(target)
    scopes.add(scope)
    targets = scope_to_targets.get(scope)
    if targets is None:
        targets = []
        scope_to_targets[scope] = targets
    targets.append(class_name)

def _get_scope(fqn):
    '''
    Parses and returns the scope in `fqn`.
    Example: if `fqn` is "NS1::NS2::C", "NS1::NS2" is returned.

    Args:
        fqn (string): fully-qualified class name.

    Returns:
        string: scope of `fqn`.
    '''

    pos = _find_namespace_end(fqn)
    if pos < 0: # global namespace
        return ''
    else:
        return fqn[:pos]

def _split_scope_and_class(fqn):
    '''
    Parses and returns the scope and class name in `fqn`.
    Example: if `fqn` is "NS1::NS2::C", "NS1::NS2" and "C" are returned.

    Args:
        fqn (string): fully-qualified class name.

    Returns:
        tuple[string,string]: tuple with the scope of and class name of `fqn`.
    '''

    pos = _find_namespace_end(fqn)
    if pos < 0: # global namespace
        return '', fqn
    else:
        return fqn[:pos], fqn[pos+2:]

def _find_namespace_end(fqn):
    '''
    Find the position where the namespace in `fqn` ends.

    Args:
        fqn (string): fully-qualified class name.

    Returns:
        integer: position where the namespace in `fqn` ends, i.e. the position
            of the last '::', or -1 if there is no '::' in `fqn`.
    '''

    last_found = -1
    prev_c = ''
    pos = 0
    for c in fqn:
        if c == ':' and prev_c == ':':
            last_found = pos - 1
        elif c == '<':
            # If we found a template, this is already the class name,
            # so we're done!
            break
        prev_c = c
        pos += 1

    return last_found

def _check_num_pars(f):
    '''
    Checks the number of parameters of the `f` function.

    Args:
        f (function): user pythonizor function.

    Returns:
        int: number of positional parameters of `f`.
    '''

    if sys.version_info >= (3, 0):
        npars = len(inspect.getfullargspec(f).args)
    else:
        npars = len(inspect.getargspec(f).args)

    if npars == 0 or npars > 2:
        raise TypeError("Pythonizor function {} has a wrong number of "
                        "parameters ({}). Allowed parameters are the class to "
                        "be pythonized and (optionally) its name."
                        .format(f.__name__, npars))

    return npars

def _register_pythonizations():
    '''
    Registers the ROOT pythonizations with cppyy for lazy injection.
    '''

    exclude = [ '_rdf_utils' ]
    for _, module_name, _ in  pkgutil.walk_packages(__path__):
        if module_name not in exclude:
            module = importlib.import_module(__name__ + '.' + module_name)
