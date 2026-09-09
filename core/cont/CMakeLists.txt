# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

set_property(TARGET Core APPEND PROPERTY DICT_HEADERS
  ROOT/TSeq.hxx
  ROOT/TArrayC.h
  ROOT/TArrayD.h
  ROOT/TArrayF.h
  ROOT/TArray.h
  ROOT/TArrayI.h
  ROOT/TArrayL64.h
  ROOT/TArrayL.h
  ROOT/TArrayS.h
  ROOT/TBits.h
  ROOT/TBtree.h
  ROOT/TClassTable.h
  ROOT/TClonesArray.h
  ROOT/TCollection.h
  ROOT/TCollectionProxyInfo.h
  ROOT/TExMap.h
  ROOT/THashList.h
  ROOT/THashTable.h
  ROOT/TIterator.h
  ROOT/TList.h
  ROOT/TMap.h
  ROOT/TObjArray.h
  ROOT/TObjectTable.h
  ROOT/TOrdCollection.h
  ROOT/TRefArray.h
  ROOT/TRefTable.h
  ROOT/TSeqCollection.h
  ROOT/TSortedList.h
  ROOT/TVirtualCollectionProxy.h
)

target_sources(Core PRIVATE
  src/TArrayC.cxx
  src/TArray.cxx
  src/TArrayD.cxx
  src/TArrayF.cxx
  src/TArrayI.cxx
  src/TArrayL64.cxx
  src/TArrayL.cxx
  src/TArrayS.cxx
  src/TBits.cxx
  src/TBtree.cxx
  src/TClassTable.cxx
  src/TClonesArray.cxx
  src/TCollection.cxx
  src/TExMap.cxx
  src/THashList.cxx
  src/THashTable.cxx
  src/TIterator.cxx
  src/TList.cxx
  src/TMap.cxx
  src/TObjArray.cxx
  src/TObjectTable.cxx
  src/TOrdCollection.cxx
  src/TRefArray.cxx
  src/TRefTable.cxx
  src/TSeqCollection.cxx
  src/TSortedList.cxx
)

target_include_directories(Core
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT>) # bw compatibility

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)
