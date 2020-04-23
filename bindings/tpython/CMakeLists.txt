# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT bindings/pyroot_experimental package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTPython
  HEADERS
    TPyArg.h
    TPyReturn.h
    TPython.h
  SOURCES
    src/TPyArg.cxx
    src/TPyClassGenerator.cxx
    src/TPyReturn.cxx
    src/TPython.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
  LIBRARIES
    cppyy${PYTHON_UNDER_VERSION_STRING}
)

# Disables warnings originating from deprecated register keyword in Python
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(ROOTTPython PRIVATE -Wno-register)
endif()
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" AND CMAKE_CXX_STANDARD GREATER_EQUAL 11)
    target_compile_options(ROOTTPython PRIVATE -Wno-register)
    target_compile_options(ROOTTPython PRIVATE -Wno-deprecated-register)
endif()
