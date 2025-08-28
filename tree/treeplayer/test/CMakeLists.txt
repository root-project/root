# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(treeplayer_array array.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(treeplayer_basic basic.cxx LIBRARIES TreePlayer)

if(NOT MSVC OR win_broken_tests) 
# disable this test on Windows since it complains about pointer address
ROOT_ADD_GTEST(treeplayer_branchobject branchobject.cxx LIBRARIES TreePlayer)
add_custom_command(TARGET treeplayer_branchobject POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/branchobject.root branchobject.root)
endif()

ROOT_ADD_GTEST(treeplayer_gh16804 gh16804.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(treeplayer_gh16805 gh16805.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(treeplayer_leafs leafs.cxx LIBRARIES TreePlayer)
add_custom_command(TARGET treeplayer_leafs POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data.h data.h)

ROOT_ADD_GTEST(treeplayer_readerarray_iterator readerarray_iterator.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(treeplayer_regressions regressions.cxx LIBRARIES TreePlayer Physics)

ROOT_ADD_GTEST(ttreeindex_clone ttreeindex_clone.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(ttreeindex_getlistoffriends ttreeindex_getlistoffriends.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(ttreereader_friends ttreereader_friends.cxx LIBRARIES TreePlayer)

if(imt)
   if (NOT MSVC)
      ROOT_ADD_GTEST(treeprocessors treeprocs/treeprocessors.cxx LIBRARIES TreePlayer)
   endif()
   if(xrootd)
      ROOT_ADD_GTEST(treeprocessormt_remotefiles treeprocs/treeprocessormt_remotefiles.cxx LIBRARIES TreePlayer)
   endif()
endif()


