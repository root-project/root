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


def _kwargs_to_roocmdargs(*args, **kwargs):
    """Helper function to check kwargs with keys that correspond to a function that creates RooCmdArg."""

    def getter(k, v):
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

        try:
            # If the keyword argument value is a tuple, list, set, or dict, first
            # try to unpack it as parameters to the RooCmdArg-generating
            # function. If this doesn't succeed, the tuple, list, or dict,
            # will be passed directly to the function as it's only argument.
            if isinstance(v, (tuple, list, set)):
                return func(*v)
            elif isinstance(v, (dict,)):
                return func(**v)
        except:
            pass

        return func(v)

    if kwargs:
        args = args + tuple((getter(k, v) for k, v in kwargs.items()))
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
                # Here getattr(ROOT, value) would have less overhead, but it's
                # also less convenient. With `eval`, the string that is passed
                # by the user can also contain postfix operations on the enum
                # value, which is often used for columns, e.g. `kGreen+1`.
                return eval("ROOT." + value)
            except:
                raise ValueError(
                    "Unsupported value passed. The value either has to be the name of an attribute of the ROOT module, or match with one of the following values that get translated to ROOT attributes: {}".format(
                        lookup_map
                    )
                )
    else:
        return value


def _dict_to_std_map(arg_dict, allowed_val_dict):
    """
    Helper function to convert python dict to std::map.

    :param arg_dict: Python Dictionary passed to convert into std::map
    :param allowed_val_dict: Contains the instances in the form of list or string allowed for the function, to check if passed dict is valid
    :return: std::map
    """
    # Default bindings: bmix.plotOn(frame2, ROOT.RooFit.Slice(tagFlav, "B0"), ROOT.RooFit.Slice(mixState, "mixed"))
    # With pythonizations: bmix.plotOn(frame2, Slice={tagFlav: "B0", mixState: "mixed"})

    import ROOT

    def all_of_class(d, type, check_key):
        return all([isinstance(key if check_key else value, type) for key, value in d.items()])

    def get_python_class(cpp_type_name):

        cpp_type_name = cpp_type_name.replace("*", "")

        if cpp_type_name in ["std::string", "string"]:
            return str
        if cpp_type_name == "int":
            return int

        # otherwise try to get class from the ROOT namespace
        return getattr(ROOT, cpp_type_name)

    def prettyprint_str_list(l):
        if len(l) == 1:
            return l[0]
        if len(l) == 2:
            return l[0] + " or " + l[1]
        return ", ".join(l[:-1]) + ", or " + l[-1]

    def get_template_args(import_dict):

        key_type = None
        value_type = None

        def get_python_typenames(typenames):
            return [get_python_class(t).__name__ for t in typenames]

        for key_typename in allowed_val_dict.keys():
            if all_of_class(import_dict, get_python_class(key_typename), True):
                key_type = key_typename

                if type(allowed_val_dict[key_typename]) == str:
                    allowed_val_dict[key_typename] = [allowed_val_dict[key_typename]]

                for val_typename in allowed_val_dict[key_typename]:
                    if all_of_class(import_dict, get_python_class(val_typename), False):
                        value_type = val_typename

                if value_type is None:
                    raise TypeError(
                        "All dictionary values must be of the same type, which can be either "
                        + prettyprint_str_list(get_python_typenames(allowed_val_dict[key_typename]))
                        + ", given the key type "
                        + get_python_class(key_type).__name__
                        + "."
                    )

        if key_type is None:
            raise TypeError(
                "All dictionary keys must be of the same type, which can be either "
                + prettyprint_str_list(get_python_typenames(allowed_val_dict))
                + "."
            )

        return key_type + "," + value_type

    # The std::map created by this function usually contains non-owning pointers as values.
    # This is not considered by Pythons reference counter. To ensure that the pointed-to objects
    # live at least as long as the std::map, a python list containing references to these objects
    # is added as an attribute to the std::map.
    arg_map = ROOT.std.map[get_template_args(arg_dict)]()
    arg_map.keepalive = list()
    for key, value in arg_dict.items():
        arg_map.keepalive.append(value)
        arg_map[key] = value

    return arg_map


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


def cpp_signature(sig):
    """Decorator to set the `_cpp_signature` attribute of a function.
    This information can be used to generate the documentation.
    """

    def decorator(func):
        func._cpp_signature = sig
        return func

    return decorator
