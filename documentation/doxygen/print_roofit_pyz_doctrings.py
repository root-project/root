# -------------------------------------------------------------------------------
#  Author: Jonas Rembser <jonas.rembser@cern.ch> CERN
# -------------------------------------------------------------------------------

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

# Creates the doxygen documentation for the RooFit pythonizations, which includes:
#   - a PyROOT box for each pythonized class or member function
#   - a separate page for the RoofitPythonizations group where all the RooFit
#     pythonization documentation is aggregated

import inspect


def clean_cpp_signature(sig):
    """Clean everything from the C++ signature that prohibits doxygen from automatically creating the correct link."""

    def strip_defaults_from_param_sig(param_sig):
        # strip default parameter values
        split_was_at_equal_sign = [False]
        for c in param_sig:
            if c == "=":
                split_was_at_equal_sign.append(True)
            elif c == ",":
                split_was_at_equal_sign.append(False)
        l = param_sig.replace("=", ",").split(",")
        l = [l for l, was in zip(l, split_was_at_equal_sign) if not was]
        return ",".join(l)

    def strip_defaults(sig):
        pbegin = sig.index("(") + 1
        pend = sig.rindex(")")
        param_sig = sig[pbegin:pend]
        return sig[:pbegin] + strip_defaults_from_param_sig(param_sig) + sig[pend:]

    def strip_output(sig):
        beg = sig.index("(")
        tmp = sig[:beg].replace("*", "").replace("&", "")
        return tmp.strip().split(" ")[-1] + sig[beg:]

    # replace new lines
    sig = sig.replace("\n", " ")

    # remove semicolons
    sig = sig.replace(";", "")

    # remove default parameters in the signature
    sig = strip_defaults(sig)

    # remove output parameter from signature
    sig = strip_output(sig)

    # remove double whitespaces
    while "  " in sig:
        sig = sig.replace("  ", " ")

    # return processed signature with whitespaces stripped from beginning and end
    return sig.strip()


def write_pyroot_block_for_class(klass):

    if klass.__doc__ is None:
        return

    print("\class " + klass.__name__)
    print("\\brief \parblock \endparblock")
    print("\htmlonly")
    print('<div class="pyrootbox">')
    print("\endhtmlonly")
    print("## PyROOT")

    print(inspect.cleandoc(klass.__doc__))

    print("\htmlonly")
    print("</div>")
    print("\endhtmlonly")
    print("")


def write_pyroot_block_for_function(func):

    if func.__doc__ is None or not hasattr(func, "_cpp_signature"):
        return

    sigs = func._cpp_signature
    if isinstance(sigs, str):
        sigs = [sigs]

    for sig in sigs:
        print("\\fn " + clean_cpp_signature(sig))
        print("\\brief \parblock \endparblock")
        print("\htmlonly")
        print('<div class="pyrootbox">')
        print("\endhtmlonly")
        print("## PyROOT")

        print(inspect.cleandoc(func.__doc__))

        print("\htmlonly")
        print("</div>")
        print("\endhtmlonly")
        print("")


def print_roofit_pythonization_page():
    """Prints the doxygen code for the RooFit pythonization page."""
    from ROOT._pythonization import _roofit

    def member_funcs_have_doc(python_class):
        funcs_have_doc = False
        for func_name in _roofit.get_defined_attributes(python_klass):
            if not getattr(python_class, func_name).__doc__ is None:
                funcs_have_doc = True
        return funcs_have_doc

    # Fill separate RooFit pythonization page, starting with the introduction and table of contents...
    print("\defgroup RoofitPythonizations Roofit pythonizations")
    print("\ingroup Roofitmain")
    for python_klass in _roofit.python_classes:
        if python_klass.__doc__ is None and not member_funcs_have_doc(python_klass):
            continue
        class_name = python_klass.__name__
        print("- [" + class_name + "](\\ref _" + class_name.lower() + ")")

        for func_name in _roofit.get_defined_attributes(python_klass):
            func = getattr(python_klass, func_name)
            if func.__doc__ is None:
                continue
            print("  - [" + func.__name__ + "](\\ref _" + (python_klass.__name__ + "_" + func.__name__).lower() + ")")

    print("")

    # ...and then iterating over all pythonized classes and functions
    for python_klass in _roofit.python_classes:

        if python_klass.__doc__ is None and not member_funcs_have_doc(python_klass):
            continue

        print("\\anchor _" + python_klass.__name__.lower())
        print("## " + python_klass.__name__)
        print("\see " + python_klass.__name__)
        if not python_klass.__doc__ is None:
            print("")
            print(inspect.cleandoc(python_klass.__doc__))
        print("")

        for func_name in _roofit.get_defined_attributes(python_klass):
            func = getattr(python_klass, func_name)
            if func.__doc__ is None:
                continue
            print("\\anchor _" + (python_klass.__name__ + "_" + func.__name__).lower())
            print("### " + python_klass.__name__ + "." + func.__name__)
            print(inspect.cleandoc(func.__doc__))
            print("")
            if hasattr(func, "_cpp_signature"):
                sigs = func._cpp_signature
                if isinstance(sigs, str):
                    sigs = [sigs]
                for sig in sigs:
                    print("\see " + clean_cpp_signature(sig))
                    print("")


def print_pyroot_blocks_for_cpp_docs():
    """Print PyROOT blocks for the RooFit C++ documentation."""
    from ROOT._pythonization import _roofit

    for python_klass in _roofit.python_classes:

        write_pyroot_block_for_class(python_klass)

        func_names = _roofit.get_defined_attributes(python_klass)

        for func_name in func_names:
            func = getattr(python_klass, func_name)
            write_pyroot_block_for_function(func)

    for python_function in _roofit.python_roofit_functions:
        write_pyroot_block_for_function(python_function)


if __name__ == "__main__":
    try:
        from ROOT._pythonization import _roofit
    except ModuleNotFoundError as err:
        import sys
        print('ROOT PYTHONPATH not set, or roofit not installed', file=sys.stderr)
        print(err, file=sys.stderr)
        sys.exit()

    print("/**")
    print_roofit_pythonization_page()
    print("")
    print_pyroot_blocks_for_cpp_docs()
    print("*/")
