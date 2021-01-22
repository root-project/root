# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

file(COPY test_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY test_shared_libraries DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ROOT_ADD_PYUNITTEST(distrdf_integration_local_test_dataframe_asnumpy test_dataframe_asnumpy.py)
ROOT_ADD_PYUNITTEST(distrdf_integration_local_test_dataframe_histograms test_dataframe_histograms.py)
ROOT_ADD_PYUNITTEST(distrdf_integration_local_test_dataframe_misc test_dataframe_misc.py)
ROOT_ADD_PYUNITTEST(distrdf_integration_local_test_dataframe_snapshot test_dataframe_snapshot.py)
ROOT_ADD_PYUNITTEST(distrdf_integration_local_test_headers_include test_headers_include.py)
ROOT_ADD_PYUNITTEST(distrdf_integration_local_test_shared_lib_include test_shared_lib_include.py)
