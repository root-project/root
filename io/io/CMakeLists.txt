############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

set(headers ROOT/TBufferMerger.hxx
   TArchiveFile.h
   TBufferFile.h
   TBufferText.h
   TBufferIO.h
   TBufferJSON.h
   TCollectionProxyFactory.h
   TContainerConverters.h
   TEmulatedMapProxy.h
   TEmulatedCollectionProxy.h
   TDirectoryFile.h
   TFileCacheRead.h
   TFileMerger.h
   TFree.h
   TFileCacheWrite.h
   TFilePrefetch.h
   TFile.h
   TFPBlock.h
   TGenCollectionStreamer.h
   TGenCollectionProxy.h
   TKey.h
   TKeyMapFile.h
   TLockFile.h
   TMemFile.h
   TMapFile.h
   TMakeProject.h
   TStreamerInfoActions.h
   TVirtualCollectionIterators.h
   TStreamerInfo.h
   TVirtualObject.h
   TVirtualArray.h
   TZIPFile.h
)

set(sources src/TArchiveFile.cxx
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
)

set(headersv7 v7/inc/ROOT/TFile.hxx)

set(sourcesv7 v7/src/TFile.cxx)

include_directories(${CMAKE_SOURCE_DIR}/core/clib/res ${CMAKE_SOURCE_DIR}/io/io/res)

ROOT_GENERATE_DICTIONARY(G__RIO ${headers} STAGE1 MODULE RIO LINKDEF LinkDef.h DEPENDENCIES Core Thread)

if(root7)
   ROOT_OBJECT_LIBRARY(RIOObjs G__RIO.cxx ${sources} ${sourcesv7})
else()
   ROOT_OBJECT_LIBRARY(RIOObjs G__RIO.cxx ${sources})
endif()

ROOT_LINKER_LIBRARY(RIO $<TARGET_OBJECTS:RIOObjs> $<TARGET_OBJECTS:RootPcmObjs>
                               LIBRARIES ${CMAKE_DL_LIBS}
                               DEPENDENCIES Core Thread)

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)
