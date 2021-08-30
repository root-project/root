# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT net/netx package
# @author Pere Mato, CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(Netx
  HEADERS
    TXNetFile.h
    TXNetFileStager.h
    TXNetSystem.h
  SOURCES
    src/TXNetFile.cxx
    src/TXNetFileStager.cxx
    src/TXNetSystem.cxx
  DEPENDENCIES
    Net
    RIO
    Thread
  BUILTINS
    XROOTD
)

target_compile_options(Netx PRIVATE ${XROOTD_CFLAGS})
target_include_directories(Netx PRIVATE ${XROOTD_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/proof/proofd/inc)
target_link_libraries(Netx PRIVATE ${XROOTD_LIBRARIES})
