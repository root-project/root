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
ROOT_ADD_GTEST(dataframe_regression dataframe_regression.cxx LIBRARIES Physics ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_utils dataframe_utils.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_report dataframe_report.cxx LIBRARIES ROOTDataFrame)

if(MSVC)
  set_property(TARGET dataframe_cache APPEND_STRING PROPERTY LINK_FLAGS " -STACK:10000000")
  set_property(TARGET dataframe_interface APPEND_STRING PROPERTY LINK_FLAGS " -STACK:10000000")
endif()

ROOT_ADD_GTEST(dataframe_splitcoll_arrayview dataframe_splitcoll_arrayview.cxx LIBRARIES ROOTDataFrame)
target_include_directories(dataframe_splitcoll_arrayview PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ROOT_GENERATE_DICTIONARY(TwoFloatsDict TwoFloats.h MODULE dataframe_splitcoll_arrayview LINKDEF TwoFloatsLinkDef.h OPTIONS -inlineInputHeader)

ROOT_ADD_GTEST(dataframe_redefine dataframe_redefine.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_definepersample dataframe_definepersample.cxx LIBRARIES ROOTDataFrame)

if(NOT MSVC OR win_broken_tests)
  ROOT_ADD_GTEST(dataframe_simple dataframe_simple.cxx LIBRARIES ROOTDataFrame)
  target_include_directories(dataframe_simple PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  ROOT_GENERATE_DICTIONARY(DataFrameSimpleDict MaxSlotHelper.h SimpleFiller.h MODULE dataframe_simple
                           LINKDEF DataFrameSimpleLinkDef.h OPTIONS -inlineInputHeader DEPENDENCIES ROOTDataFrame Hist)
endif()
ROOT_ADD_GTEST(dataframe_vecops dataframe_vecops.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_helpers dataframe_helpers.cxx LIBRARIES ROOTDataFrame)

if(NOT (MSVC OR (APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES arm64)) OR win_broken_tests OR M1_BROKEN_TESTS)
  ROOT_ADD_GTEST(dataframe_snapshot dataframe_snapshot.cxx LIBRARIES ROOTDataFrame)
endif()

ROOT_ADD_GTEST(dataframe_datasetspec dataframe_datasetspec.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_display dataframe_display.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_ranges dataframe_ranges.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_leaves dataframe_leaves.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_resptr dataframe_resptr.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_take dataframe_take.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_entrylist dataframe_entrylist.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_merge_results dataframe_merge_results.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_samplecallback dataframe_samplecallback.cxx CounterHelper.h LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(dataframe_vary dataframe_vary.cxx LIBRARIES ROOTDataFrame)

#### TESTS FOR DIFFERENT DATASOURCES ####
if(MSVC AND MSVC_VERSION GREATER_EQUAL 1925 AND MSVC_VERSION LESS 1929 OR CMAKE_CXX_STANDARD LESS 17)
  # TODO: remove this workaround for MS compiler bug #1441527 once fixed
  string(REPLACE "-Od -Z7" "-O2" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  string(REPLACE "-Z7" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif()

ROOT_ADD_GTEST(datasource_more datasource_more.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(datasource_root datasource_root.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(datasource_trivial datasource_trivial.cxx LIBRARIES ROOTDataFrame)
ROOT_ADD_GTEST(datasource_lazy datasource_lazy.cxx LIBRARIES ROOTDataFrame)
configure_file(RCsvDS_test_headers.csv . COPYONLY)
configure_file(RCsvDS_test_noheaders.csv . COPYONLY)
configure_file(RCsvDS_test_empty.csv . COPYONLY)
configure_file(RCsvDS_test_win.csv . COPYONLY)
configure_file(RCsvDS_test_NaNs.csv . COPYONLY)
ROOT_ADD_GTEST(datasource_csv datasource_csv.cxx LIBRARIES ROOTDataFrame)
if(MSVC AND "${CMAKE_GENERATOR_PLATFORM}" MATCHES "x64")
  set_source_files_properties(dataframe_vary.cxx COMPILE_FLAGS "-bigobj")
endif()

#### TESTS REQUIRING EXTRA ROOT FEATURES ####
if (imt)
   ROOT_ADD_GTEST(dataframe_concurrency dataframe_concurrency.cxx LIBRARIES ROOTDataFrame)
endif()

if(ARROW_FOUND)
  ROOT_ADD_GTEST(datasource_arrow datasource_arrow.cxx LIBRARIES ROOTDataFrame ${ARROW_SHARED_LIB})
  target_include_directories(datasource_arrow BEFORE PRIVATE ${ARROW_INCLUDE_DIR})
endif()

if(root7)
  ROOT_ADD_GTEST(datasource_ntuple datasource_ntuple.cxx LIBRARIES ROOTDataFrame)
endif()

if(sqlite)
  configure_file(RSqliteDS_test.sqlite . COPYONLY)
  ROOT_ADD_GTEST(datasource_sqlite datasource_sqlite.cxx LIBRARIES ROOTDataFrame ${SQLITE_LIBRARIES})
  target_include_directories(datasource_sqlite BEFORE PRIVATE ${SQLITE_INCLUDE_DIR})
endif()

#### PYTHON TESTS ####
if(pyroot)
  if(NOT MSVC OR win_broken_tests OR CMAKE_CXX_STANDARD GREATER 14)
    ROOT_ADD_PYUNITTEST(dataframe_misc dataframe_misc.py)
    ROOT_ADD_PYUNITTEST(dataframe_histograms dataframe_histograms.py)
    ROOT_ADD_PYUNITTEST(dataframe_cache dataframe_cache.py ENVIRONMENT LEGACY_PYROOT=${pyroot_legacy})
  endif()
  if(NOT MSVC OR win_broken_tests)
    ROOT_ADD_PYUNITTEST(dataframe_merge_results dataframe_merge_results.py)
  endif()
endif()
