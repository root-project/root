# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

###########################################################
# CMakeLists.txt file for building TPython
###########################################################

set(TPYTHON_HEADERS
  TPyArg.h
  TPyClassGenerator.h
  TPyReturn.h
  TPython.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTTPython
  HEADERS
    ${TPYTHON_HEADERS}
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
    cpyrt
    # We link libTPython against Python libraries to compensate for the fact that libcppyy
    # is built with unresolved symbols. If we didn't do this, invoking TPython from C++
    # would not work.
    Python3::Python
)

set(RELATIVE_INC_HEADERS ${TPYTHON_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTTPython
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

# On Windows we can't use the stable ABI yet: it requires linking against a
# different libpython, so as long as we don't build all translation units in
# the ROOT Pythonization library with the stable ABI we should not use it.
# Free-threaded (Py_GIL_DISABLED) interpreters don't support the limited API.
if(NOT MSVC AND NOT Python3_GIL_DISABLED)
  target_compile_options(ROOTTPython PRIVATE -DPy_LIMITED_API=0x030A0000)
endif()

if(MSVC)
  target_link_options(ROOTTPython PRIVATE "-ignore:4197")
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
