# Author: Stephan Hageboeck, CERN 04/2020

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################


from ._utils import _kwargs_to_roocmdargs, cpp_signature


def make_json_for_variable(var_name, var_dict):
    val = None
    max_value = None
    min_value = None
    is_constant = None
    if "value" in var_dict:
        val = var_dict["value"]
        is_constant = True
    if ("max" in var_dict) and ("min" in var_dict):
        max_value = var_dict["max"]
        min_value = var_dict["min"]
        is_constant = False
    if is_constant is None:
        raise ValueError(
            "Invalid Syntax: Please provide either 'value' or 'min' and 'max' or both"
        )
    else:
        if not is_constant:
            val = (max_value + min_value) / 2
        # Create dictionary with domains and parameter points
        json_dict = {
            "domains": [
                {
                    "axes": [{"name": var_name}],
                    "name": "default_domain",
                    "type": "product_domain",
                }
            ],
            "parameter_points": [
                {
                    "name": "default_values",
                    "parameters": [{"name": var_name, "value": val}],
                }
            ],
        }
        if is_constant:
            json_dict["parameter_points"][0]["parameters"][0]["const"] = True
            json_dict["misc"] = {"ROOT_internal": {var_name: {"tags": "Constant"}}}
        if not is_constant:
            json_dict["domains"][0]["axes"][0]["max"] = max_value
            json_dict["domains"][0]["axes"][0]["min"] = min_value
        return json_dict


class RooWorkspace(object):
    r"""The RooWorkspace::import function can't be used in PyROOT because `import` is a reserved python keyword.
    For this reason, an alternative with a capitalized name is provided:
    \code{.py}
    workspace.Import(x)
    \endcode
    """

    @cpp_signature(
        "Bool_t RooWorkspace::import(const RooAbsArg& arg,"
        "    const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),"
        "    const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),"
        "    const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;"
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
                    expr = (
                        value[0:parenthesis_index]
                        + "::"
                        + key
                        + value[parenthesis_index:]
                    )
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

            # Add name attribute to the dictionary, and create the JSON string for the object's JSONTree
            is_variable = not "type" in value
            if is_variable:
                # Import variable
                json_dict = make_json_for_variable(key, value)
                json_string = json.dumps(json_dict, separators=(",", ":"))
                ROOT.RooJSONFactoryWSTool(self).importVarfromString(json_string)
            else:
                # Imports functions/p.d.f.s
                value["name"] = key
                json_string = json.dumps(value, separators=(",", ":"))
                ROOT.RooJSONFactoryWSTool(self).importFunction(json_string, False)
        else:
            raise TypeError(
                "Object of type 'str' or 'dict' expected but "
                + type(value)
                + " was given"
            )

    @cpp_signature(
        [
            "Bool_t RooWorkspace::import(const RooAbsArg& arg,"
            "         const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),"
            "         const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),"
            "         const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;",
            "Bool_t RooWorkspace::import(RooAbsData& data,"
            "         const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),"
            "         const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),"
            "         const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;",
            "RooWorkspace::import(const char *fileSpec,"
            "         const RooCmdArg& arg1=RooCmdArg(),const RooCmdArg& arg2=RooCmdArg(),const RooCmdArg& arg3=RooCmdArg(),"
            "         const RooCmdArg& arg4=RooCmdArg(),const RooCmdArg& arg5=RooCmdArg(),const RooCmdArg& arg6=RooCmdArg(),"
            "         const RooCmdArg& arg7=RooCmdArg(),const RooCmdArg& arg8=RooCmdArg(),const RooCmdArg& arg9=RooCmdArg()) ;",
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
            raise AttributeError(
                'Resetting the "'
                + name
                + '" attribute of a RooWorkspace is not allowed!'
            )
        object.__setattr__(self, name, value)


def RooWorkspace_import(self, *args, **kwargs):
    r"""The RooWorkspace::import function can't be used in PyROOT because `import` is a reserved python keyword.
    So, Import() is used and pythonized with the command argument pythonization.
    The keywords must correspond to the CmdArg of the `import()` function.
    """
    # Redefinition of `RooWorkspace.import()` for keyword arguments.
    args, kwargs = _kwargs_to_roocmdargs(*args, **kwargs)
    return self._import(*args, **kwargs)


setattr(RooWorkspace, "import", RooWorkspace_import)
