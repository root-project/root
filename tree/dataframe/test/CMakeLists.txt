# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN, Enrico Guiraud CERN

#### C++ TESTS ####
ROOT_ADD_GTEST(dataframe_friends dataframe_friends.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_colnames dataframe_colnames.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_cache dataframe_cache.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_callbacks dataframe_callbacks.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_histomodels dataframe_histomodels.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_interface dataframe_interface.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_nodes dataframe_nodes.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_regression dataframe_regression.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_simple dataframe_simple.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_snapshot dataframe_snapshot.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_utils dataframe_utils.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_report dataframe_report.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_helpers dataframe_helpers.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_ranges dataframe_ranges.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_leaves dataframe_leaves.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_vecops dataframe_vecops.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_resptr dataframe_resptr.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_take dataframe_take.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_entrylist dataframe_entrylist.cxx LIBRARIES ROOTDataFrame)

if (imt)
   ROOT_ADD_GTEST(dataframe_concurrency dataframe_concurrency.cxx LIBRARIES ROOTDataFrame)
endif()

ROOT_ADD_GTEST(datasource_more datasource_more.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(datasource_root datasource_root.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(datasource_trivial datasource_trivial.cxx LIBRARIES ROOTDataFrame)
configure_file(RCsvDS_test_headers.csv . COPYONLY)
configure_file(RCsvDS_test_noheaders.csv . COPYONLY)
if(ARROW_FOUND)
  ROOT_ADD_GTEST(datasource_arrow datasource_arrow.cxx LIBRARIES ROOTDataFrame ${ARROW_SHARED_LIB})
  target_include_directories(datasource_arrow BEFORE PRIVATE ${ARROW_INCLUDE_DIR})
endif()
if(sqlite)
  configure_file(RSqliteDS_test.sqlite . COPYONLY)
  ROOT_ADD_GTEST(datasource_sqlite datasource_sqlite.cxx LIBRARIES ROOTDataFrame ${SQLITE_LIBRARIES})
  target_include_directories(datasource_sqlite BEFORE PRIVATE ${SQLITE_INCLUDE_DIR})
endif()
ROOT_ADD_GTEST(datasource_csv datasource_csv.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(datasource_lazy datasource_lazy.cxx LIBRARIES ROOTDataFrame)

#### PYTHON TESTS ####
ROOT_ADD_PYUNITTEST(dataframe_misc dataframe_misc.py)
ROOT_ADD_PYUNITTEST(dataframe_histograms dataframe_histograms.py)
ROOT_ADD_PYUNITTEST(dataframe_cache dataframe_cache.py ENVIRONMENT EXP_PYROOT=${pyroot_experimental})
