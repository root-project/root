# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Don't bother testing the distributed layer of RDataFrame if it's not available
# in the build. Also don't test on windows
if (dataframe AND NOT MSVC)

ROOT_ADD_PYUNITTEST(distrdf_unit_test_buildranges test_buildranges.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_test_callable_generator test_callable_generator.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_test_friendinfo test_friendinfo.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_test_headnode test_headnode.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_test_node test_node.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_test_operation test_operation.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_test_proxy test_proxy.py)

endif()
