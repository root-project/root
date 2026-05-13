# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/multiproc package
############################################################################

if(WIN32)
  return()
endif()


set(MULTIPROC_HEADERS
  MPCode.h
  MPSendRecv.h
  PoolUtils.h
  TMPClient.h
  TMPWorker.h
  TMPWorkerExecutor.h
  TProcPool.h
  ROOT/TProcessExecutor.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(MultiProc STAGE1
  HEADERS
    ${MULTIPROC_HEADERS}
  SOURCES
    src/MPSendRecv.cxx
    src/TMPClient.cxx
    src/TMPWorker.cxx
    src/TProcessExecutor.cxx
  LIBRARIES
    ${CMAKE_DL_LIBS}
  DEPENDENCIES
    Core
    Net
)

set(RELATIVE_INC_HEADERS ${MULTIPROC_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

if(NOT CMAKE_VERSION VERSION_LESS "3.23.0")
  target_sources(
      MultiProc
      PRIVATE
      FILE_SET private_header_files
      TYPE HEADERS
      BASE_DIRS inc/
      FILES
          ${RELATIVE_INC_HEADERS}
          inc/LinkDef.h
  )
endif()
