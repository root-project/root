# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT ntuple package
# @author Jakob Blomer CERN
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTNTuple
HEADERS
  ROOT/RCluster.hxx
  ROOT/RClusterPool.hxx
  ROOT/RColumn.hxx
  ROOT/RColumnElementBase.hxx
  ROOT/REntry.hxx
  ROOT/RField.hxx
  ROOT/RFieldBase.hxx
  ROOT/RFieldToken.hxx
  ROOT/RFieldUtils.hxx
  ROOT/RFieldVisitor.hxx
  ROOT/RMiniFile.hxx
  ROOT/RNTuple.hxx
  ROOT/RNTupleDescriptor.hxx
  ROOT/RNTupleFillContext.hxx
  ROOT/RNTupleFillStatus.hxx
  ROOT/RNTupleImtTaskScheduler.hxx
  ROOT/RNTupleJoinTable.hxx
  ROOT/RNTupleMerger.hxx
  ROOT/RNTupleMetrics.hxx
  ROOT/RNTupleModel.hxx
  ROOT/RNTupleParallelWriter.hxx
  ROOT/RNTupleProcessor.hxx
  ROOT/RNTupleRange.hxx
  ROOT/RNTupleReadOptions.hxx
  ROOT/RNTupleReader.hxx
  ROOT/RNTupleSerialize.hxx
  ROOT/RNTupleTypes.hxx
  ROOT/RNTupleUtil.hxx
  ROOT/RNTupleUtils.hxx
  ROOT/RNTupleView.hxx
  ROOT/RNTupleWriteOptions.hxx
  ROOT/RNTupleWriteOptionsDaos.hxx
  ROOT/RNTupleWriter.hxx
  ROOT/RNTupleZip.hxx
  ROOT/RPage.hxx
  ROOT/RPageAllocator.hxx
  ROOT/RPageNullSink.hxx
  ROOT/RPagePool.hxx
  ROOT/RPageSinkBuf.hxx
  ROOT/RPageStorage.hxx
  ROOT/RPageStorageFile.hxx
  ROOT/RRawPtrWriteEntry.hxx
SOURCES
  src/RCluster.cxx
  src/RClusterPool.cxx
  src/RColumn.cxx
  src/RColumnElement.cxx
  src/RField.cxx
  src/RFieldBase.cxx
  src/RFieldMeta.cxx
  src/RFieldSequenceContainer.cxx
  src/RFieldUtils.cxx
  src/RFieldVisitor.cxx
  src/RMiniFile.cxx
  src/RNTuple.cxx
  src/RNTupleDescriptor.cxx
  src/RNTupleDescriptorFmt.cxx
  src/RNTupleFillContext.cxx
  src/RNTupleJoinTable.cxx
  src/RNTupleMerger.cxx
  src/RNTupleMetrics.cxx
  src/RNTupleModel.cxx
  src/RNTupleParallelWriter.cxx
  src/RNTupleProcessor.cxx
  src/RNTupleReader.cxx
  src/RNTupleSerialize.cxx
  src/RNTupleUtils.cxx
  src/RNTupleWriteOptions.cxx
  src/RNTupleWriter.cxx
  src/RNTupleView.cxx
  src/RPage.cxx
  src/RPageAllocator.cxx
  src/RPagePool.cxx
  src/RPageSinkBuf.cxx
  src/RPageStorage.cxx
  src/RPageStorageFile.cxx
LINKDEF
  LinkDef.h
DEPENDENCIES
  Imt
  RIO
  ROOTVecOps
)

target_link_libraries(ROOTNTuple PRIVATE xxHash::xxHash)

# Enable RNTuple support for Intel DAOS
if(daos OR daos_mock)
  set(ROOTNTuple_EXTRA_HEADERS ROOT/RPageStorageDaos.hxx)
  target_sources(ROOTNTuple PRIVATE src/RDaos.cxx src/RPageStorageDaos.cxx)
  target_compile_definitions(ROOTNTuple PRIVATE R__ENABLE_DAOS)

  target_include_directories(ROOTNTuple PRIVATE ${UUID_INCLUDE_DIR})
  target_link_libraries(ROOTNTuple PRIVATE ${UUID_LIBRARY})
  if(daos_mock)
    target_include_directories(ROOTNTuple PRIVATE inc/ROOT/libdaos_mock/)
    target_sources(ROOTNTuple PRIVATE src/libdaos_mock/libdaos_mock.cxx)
  else()
    target_include_directories(ROOTNTuple PRIVATE ${DAOS_INCLUDE_DIRS})
    target_link_libraries(ROOTNTuple PRIVATE ${DAOS_LIBRARIES})
  endif()
endif()

if(MSVC)
  target_compile_definitions(ROOTNTuple PRIVATE _USE_MATH_DEFINES)
endif()

ROOT_ADD_TEST_SUBDIRECTORY(test)
