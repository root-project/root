# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT R package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(RInterface
  HEADERS
    RExports.h
    TRDataFrame.h
    TRFunctionExport.h
    TRFunctionImport.h
    TRInterface.h
    TRInternalFunction.h
    TRObject.h
  SOURCES
    src/RExports.cxx
    src/TRCompletion.cxx
    src/TRDataFrame.cxx
    src/TRFunctionExport.cxx
    src/TRFunctionImport.cxx
    src/TRInterface.cxx
    src/TRObject.cxx
  DICTIONARY_OPTIONS
    -I${R_INCLUDE_DIR}
    -I${R_Rcpp_INCLUDE_DIR}
    -I${R_RInside_INCLUDE_DIR}
  DEPENDENCIES
    Core
    Matrix
    RIO
    Thread
)

ROOT_ADD_CXX_FLAG(_R_FLAGS -Wno-cast-function-type)
ROOT_ADD_CXX_FLAG(_R_FLAGS -Wno-overloaded-virtual)

separate_arguments(_R_FLAGS)

target_compile_options(RInterface PUBLIC ${_R_FLAGS})
target_include_directories(RInterface BEFORE PUBLIC ${R_INCLUDE_DIRS})
target_link_libraries(RInterface PUBLIC ${R_LIBRARIES} readline)
