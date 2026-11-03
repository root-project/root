# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

if (WIN32)
  set(rawfile_local_headers ROOT/RRawFileWin.hxx)
  set(rawfile_local_sources src/RRawFileWin.cxx)
else ()
  set(rawfile_local_headers ROOT/RRawFileUnix.hxx)
  set(rawfile_local_sources src/RRawFileUnix.cxx)
endif ()

if (uring)
  list(APPEND rawfile_local_headers ROOT/RIoUring.hxx)
endif ()

ROOT_LINKER_LIBRARY(RIO
  src/InternalIOUtils.cxx
  src/RConcurrentHashColl.cxx
  src/RFile.cxx
  src/RRawFile.cxx
  ${rawfile_local_sources}
  src/TArchiveFile.cxx
  src/TBufferFile.cxx
  src/TBufferText.cxx
  src/TBufferIO.cxx
  src/TBufferJSON.cxx
  src/TBufferMerger.cxx
  src/TBufferMergerFile.cxx
  src/TCollectionProxyFactory.cxx
  src/TContainerConverters.cxx
  src/TEmulatedMapProxy.cxx
  src/TEmulatedCollectionProxy.cxx
  src/TDirectoryFile.cxx
  src/TFileCacheRead.cxx
  src/TFileMerger.cxx
  src/TFree.cxx
  src/TFileCacheWrite.cxx
  src/TFilePrefetch.cxx
  src/TFile.cxx
  src/TFPBlock.cxx
  src/TGenCollectionStreamer.cxx
  src/TGenCollectionProxy.cxx
  src/TKey.cxx
  src/TKeyMapFile.cxx
  src/TLockFile.cxx
  src/TMemFile.cxx
  src/TMapFile.cxx
  src/TMakeProject.cxx
  src/TStreamerInfo.cxx
  src/TStreamerInfoActions.cxx
  src/TStreamerInfoReadBuffer.cxx
  src/TStreamerInfoWriteBuffer.cxx
  src/TZIPFile.cxx
  $<TARGET_OBJECTS:RootPcmObjs>
  LIBRARIES
    ${CMAKE_DL_LIBS}
    Thread
  DEPENDENCIES
    Core
)

target_include_directories(RIO PRIVATE ${CMAKE_SOURCE_DIR}/builtins/clib/res)
# PUBLIC so that dependent libraries e.g. ROOTDataFrame may find the internal header too
target_include_directories(RIO PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/res>)
target_link_libraries(RIO PUBLIC ${ROOT_ATOMIC_LIBS})
target_link_libraries(RIO PRIVATE nlohmann_json::nlohmann_json)

if(uring)
  target_link_libraries(RIO PUBLIC uring::uring)
endif()

# keep include directory for ROOT/RSha256.hxx private
set_source_files_properties(src/RConcurrentHashColl.cxx
  PROPERTIES COMPILE_FLAGS -I${CMAKE_SOURCE_DIR}/core/foundation/res)

set(IO_HEADERS
  ROOT/RConcurrentHashColl.hxx
  ROOT/RFile.hxx
  ROOT/RRawFile.hxx
  ROOT/RRawFileTFile.hxx
  ${rawfile_local_headers}
  ROOT/TBufferMerger.hxx
  TArchiveFile.h
  TBufferFile.h
  TBufferIO.h
  TBufferJSON.h
  TBufferText.h
  TCollectionProxyFactory.h
  TContainerConverters.h
  TDirectoryFile.h
  TEmulatedCollectionProxy.h
  TEmulatedMapProxy.h
  TFPBlock.h
  TFile.h
  TFileCacheRead.h
  TFileCacheWrite.h
  TFileMerger.h
  TFilePrefetch.h
  TFree.h
  TGenCollectionProxy.h
  TGenCollectionStreamer.h
  TKey.h
  TKeyMapFile.h
  TLockFile.h
  TMakeProject.h
  TMapFile.h
  TMemFile.h
  TStreamerInfo.h
  TStreamerInfoActions.h
  TVirtualCollectionIterators.h
  TZIPFile.h
)

set(RELATIVE_INC_HEADERS ${IO_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)

target_sources(RIO
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)

ROOT_GENERATE_DICTIONARY(G__RIO
  ${IO_HEADERS}
  STAGE1
  MODULE
    RIO
  LINKDEF
    LinkDef.h
  OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Core
    Thread
)

# TStreamerInfoReadBuffer in O0 needs 6k on the stack. It is called
# recursively, quickly exhausting the stack. Prevent that by forcing
# the many scope-local vars to share their stack space / become
# registers, thanks to the optimizer.
if(MSVC)
  set_source_files_properties(src/TStreamerInfoReadBuffer.cxx COMPILE_FLAGS $<IF:$<CONFIG:Debug>,"/Od","/O2">)
else()
  set_source_files_properties(src/TStreamerInfoReadBuffer.cxx COMPILE_FLAGS "-O3")
endif()

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)
