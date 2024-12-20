# Author: Stephan Hageboeck, CERN 04/2020

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._utils import _kwargs_to_roocmdargs, cpp_signature


class RooWorkspace(object):
    r"""The RooWorkspace::import function can't be used in PyROOT because `import` is a reserved python keyword.
    For this reason, an alternative with a capitalized name is provided:
    \code{.py}
    workspace.Import(x)
    \endcode
    """

    __cpp_name__ = 'RooWorkspace'

    @cpp_signature(
        "Bool_t RooWorkspace::import(const RooAbsArg& arg,"
        "    const RooCmdArg& arg1={},const RooCmdArg& arg2={},const RooCmdArg& arg3={},"
        "    const RooCmdArg& arg4={},const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
        "    const RooCmdArg& arg7={},const RooCmdArg& arg8={},const RooCmdArg& arg9={}) ;"
    )
    def __init__(self, *args, **kwargs):
        r"""The RooWorkspace constructor is pythonized with the command argument pythonization.
        The keywords must correspond to the CmdArgs of the constructor.
        """
        # Redefinition of `RooWorkspace` constructor for keyword arguments.
        args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
        self._init(*args, **kwargs)

    def __getitem__(self, key):
        # To enable accessing objects in the RooWorkspace with dictionary-like syntax.
        # The key is passed to the general `RooWorkspace::obj()` function.
        return self.obj(key)

    def __setitem__(self, key, value):
        # Check if initialization is done with string
        if isinstance(value, str):
            if not self.obj(key):
                parenthesis_index = -1
                for i in range(0, len(value)):
                    if value[i] == "[" or value[i] == "(":
                        parenthesis_index = i
                        break
                # Initializes variables
                if value[parenthesis_index] == "[" and parenthesis_index != -1:
                    expr = key + value
                    self.factory(expr)
                # Initializes functions and p.d.f.s
                elif value[parenthesis_index] == "(" and parenthesis_index != -1:
                    expr = value[0:parenthesis_index] + "::" + key + value[parenthesis_index:]
                    self.factory(expr)
                # Else raises a Syntax error
                else:
                    raise SyntaxError("Invalid syntax")
            else:
                raise RuntimeError(
                    "ERROR importing object named "
                    + key
                    + " another instance with same name already in the workspace and no conflict resolution protocol specified"
                )
        elif isinstance(value, dict):
            import ROOT
            import json

            json_string = json.dumps(value, separators=(",", ":"))
            ROOT.RooJSONFactoryWSTool(self).importJSONElement(key, json_string)
        else:
            raise TypeError("Object of type 'str' or 'dict' expected but " + type(value) + " was given")

    @cpp_signature(
        [
            "Bool_t RooWorkspace::import(const RooAbsArg& arg,"
            "         const RooCmdArg& arg1={},const RooCmdArg& arg2={},const RooCmdArg& arg3={},"
            "         const RooCmdArg& arg4={},const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
            "         const RooCmdArg& arg7={},const RooCmdArg& arg8={},const RooCmdArg& arg9={}) ;",
            "Bool_t RooWorkspace::import(RooAbsData& data,"
            "         const RooCmdArg& arg1={},const RooCmdArg& arg2={},const RooCmdArg& arg3={},"
            "         const RooCmdArg& arg4={},const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
            "         const RooCmdArg& arg7={},const RooCmdArg& arg8={},const RooCmdArg& arg9={}) ;",
            "RooWorkspace::import(const char *fileSpec,"
            "         const RooCmdArg& arg1={},const RooCmdArg& arg2={},const RooCmdArg& arg3={},"
            "         const RooCmdArg& arg4={},const RooCmdArg& arg5={},const RooCmdArg& arg6={},"
            "         const RooCmdArg& arg7={},const RooCmdArg& arg8={},const RooCmdArg& arg9={}) ;",
        ]
    )
    def Import(self, *args, **kwargs):
        r"""
        Support the C++ `import()` as `Import()` in python
        """
        return getattr(self, "import")(*args, **kwargs)

    def __setattr__(self, name, value):
        # Many people pythonized the RooWorkspace themselves, by adding a new
        # attribute `_import` that calls getattr(self, "import") under the
        # hood. However, `_import` is now the reference to the original cppyy
        # overload, and resetting it with a wrapper around `import` would cause
        # infinite recursions! We prevent resetting any import-related function
        # here, which results in a clearer error to the user than an infinite
        # call stack involving the internal pythonization code.
        if name in ["_import", "import", "Import"]:
            raise AttributeError('Resetting the "' + name + '" attribute of a RooWorkspace is not allowed!')
        object.__setattr__(self, name, value)

    def _ipython_key_completions_(self):
        r"""
        Support tab completion for `__getitem__`, suggesting all components in
        the workspace.
        """
        return [c.GetName() for c in self.components()]


def RooWorkspace_import(self, *args, **kwargs):
    r"""The RooWorkspace::import function can't be used in PyROOT because `import` is a reserved python keyword.
    So, Import() is used and pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the `import()` function.
    """
    # Redefinition of `RooWorkspace.import()` for keyword arguments.
    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return self._import(*args, **kwargs)


setattr(RooWorkspace, "import", RooWorkspace_import)
