############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

ROOT_ADD_TEST_SUBDIRECTORY(test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)

set(headers ROOT/TSeq.hxx
   TArrayC.h
   TArrayF.h
   TArrayI.h
   TArrayL.h
   TBits.h
   TClassTable.h
   TCollection.h
   TExMap.h
   THashTable.h
   TList.h
   TObjArray.h
   TOrdCollection.h
   TRefTable.h
   TSortedList.h
   TArrayD.h
   TArray.h
   TArrayL64.h
   TArrayS.h
   TBtree.h
   TClonesArray.h
   TCollectionProxyInfo.h
   THashList.h
   TIterator.h
   TMap.h
   TObjectTable.h
   TRefArray.h
   TSeqCollection.h
   TVirtualCollectionProxy.h
)

set(sources TArrayC.cxx
   TArrayD.cxx
   TArrayI.cxx
   TArrayL.cxx
   TBits.cxx
   TClassTable.cxx
   TCollection.cxx
   THashList.cxx
   TIterator.cxx
   TMap.cxx
   TObjectTable.cxx
   TRefArray.cxx
   TSeqCollection.cxx
   TArray.cxx
   TArrayF.cxx
   TArrayL64.cxx
   TArrayS.cxx
   TBtree.cxx
   TClonesArray.cxx
   TExMap.cxx
   THashTable.cxx
   TList.cxx
   TObjArray.cxx
   TOrdCollection.cxx
   TRefTable.cxx
   TSortedList.cxx
)
set(Cont_dict_headers ${headers} PARENT_SCOPE)

ROOT_OBJECT_LIBRARY(Cont ${sources})

ROOT_INSTALL_HEADERS()
