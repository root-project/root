# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/rint package
############################################################################

set(RINT_HEADERS
  TRint.h
  TTabCom.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(Rint
  STAGE1
  HEADERS
    ${RINT_HEADERS}
  SOURCES
    src/TRint.cxx
    src/TTabCom.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
)

target_include_directories(Core PRIVATE inc)
set(RELATIVE_INC_HEADERS ${RINT_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      Rint
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/
      FILES
          ${RELATIVE_INC_HEADERS}
          inc/LinkDef.h
  )
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
