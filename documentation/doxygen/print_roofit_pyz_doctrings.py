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


def write_pyroot_block_for_class(klass):

    if not hasattr(klass, "_doxygen"):
        return

    print("\class " + klass.__name__)
    print("\\brief \parblock \endparblock")
    print("\htmlonly")
    print('<div class="pyrootbox">')
    print("\endhtmlonly")
    print("## PyROOT")

    print(inspect.cleandoc(klass._doxygen))

    print("\htmlonly")
    print("</div>")
    print("\endhtmlonly")
    print("")


def write_pyroot_block_for_member_func(func):

    if not hasattr(func, "_doxygen") or not hasattr(func, "_cpp_signature"):
        return

    print("\\fn " + func._cpp_signature)
    print("\\brief \parblock \endparblock")
    print("\htmlonly")
    print('<div class="pyrootbox">')
    print("\endhtmlonly")
    print("## PyROOT")

    print(inspect.cleandoc(func._doxygen))

    print("\htmlonly")
    print("</div>")
    print("\endhtmlonly")
    print("")


if __name__ == "__main__":

    import ROOT.pythonization as pyz

    # Fill separate RooFit pythonization page, starting with the introduction and table of contents...
    print("/**")
    print("\defgroup RoofitPythonizations")
    print("\ingroup Roofitmain")
    print("# RooFit pythonizations")
    for python_klass in pyz._roofit.python_classes:
        if not hasattr(python_klass, "_doxygen"):
            continue
        class_name = python_klass.__name__
        print("- [" + class_name + "](\\ref _" + class_name.lower() + ")")

        func_names = pyz._roofit.get_defined_attributes(python_klass)

        for func_name in func_names:
            func = getattr(python_klass, func_name)
            if not hasattr(func, "_doxygen"):
                continue
            print("  - [" + func.__name__ + "](\\ref _" + (python_klass.__name__ + "_" + func.__name__).lower() + ")")

    print("")

    # ...and then iterating over all pythonized classes and functions
    for python_klass in pyz._roofit.python_classes:
        if not hasattr(python_klass, "_doxygen"):
            continue

        print("\\anchor _" + python_klass.__name__.lower())
        print("## " + python_klass.__name__)
        print("\see " + python_klass.__name__)
        print("")
        print(inspect.cleandoc(python_klass._doxygen))
        print("")

        func_names = pyz._roofit.get_defined_attributes(python_klass)

        for func_name in func_names:
            func = getattr(python_klass, func_name)
            if not hasattr(func, "_doxygen"):
                continue
            print("\\anchor _" + (python_klass.__name__ + "_" + func.__name__).lower())
            print("### " + python_klass.__name__ + "." + func.__name__)
            print(inspect.cleandoc(func._doxygen))
            print("")
            if hasattr(func, "_cpp_signature"):
                print("\see " + func._cpp_signature)
            print("")
    print("")

    # Add PyROOT blocks to existing documentation
    for python_klass in pyz._roofit.python_classes:

        write_pyroot_block_for_class(python_klass)

        func_names = pyz._roofit.get_defined_attributes(python_klass)

        for func_name in func_names:
            func = getattr(python_klass, func_name)
            write_pyroot_block_for_member_func(func)

    print("*/")
