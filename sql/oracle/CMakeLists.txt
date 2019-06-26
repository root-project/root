# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT sql/oracle package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Oracle
  HEADERS
    TOracleResult.h
    TOracleRow.h
    TOracleServer.h
    TOracleStatement.h
  SOURCES
    src/TOracleResult.cxx
    src/TOracleRow.cxx
    src/TOracleServer.cxx
    src/TOracleStatement.cxx
  DICTIONARY_OPTIONS
    -I${ORACLE_INCLUDE_DIR}
  DEPENDENCIES
    Core
    Net
    RIO
)

# Fall back to old libstdc++ ABI if needed
if(${GLIBCXX_USE_CXX11_ABI})
  set_source_files_properties(
    src/TOracleResult.cxx
    src/TOracleRow.cxx
    src/TOracleServer.cxx
    src/TOracleStatement.cxx
    PROPERTIES
      COMPILE_DEFINITIONS _GLIBCXX_USE_CXX11_ABI=0)
endif()

target_compile_definitions(Oracle PUBLIC ${ORACLE_DEFINITIONS})
target_include_directories(Oracle PUBLIC ${ORACLE_INCLUDE_DIR})
target_link_libraries(Oracle PUBLIC ${ORACLE_LIBRARIES})
