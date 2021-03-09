# Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# Don't bother testing the distributed layer of RDataFrame if it's not available
# in the build
if (dataframe AND NOT MSVC)

file(COPY test_headers DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 2clusters.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 4clusters.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 1000clusters.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY emptytree.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY Slimmed_ntuple.root DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

ROOT_ADD_PYUNITTEST(distrdf_unit_backend_test_common test_common.py)
ROOT_ADD_PYUNITTEST(distrdf_unit_backend_test_dist test_dist.py)

# pyspark is required to run this test
if(test_distrdf_pyspark)
    ROOT_ADD_PYUNITTEST(distrdf_unit_backend_test_spark test_spark.py)
endif()

endif()
