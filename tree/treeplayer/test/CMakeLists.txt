# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

if(NOT MSVC OR win_broken_tests)
  ROOT_ADD_UNITTEST_DIR(TreePlayer Physics)
  add_custom_command(TARGET treetreeplayertestUnit POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/data.h data.h
                     COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/branchobject.root branchobject.root)
endif()

if(imt)
   if (NOT MSVC)
      ROOT_ADD_GTEST(treeprocessors treeprocs/treeprocessors.cxx LIBRARIES TreePlayer)
   endif()
   if(xrootd)
      ROOT_ADD_GTEST(treeprocessormt_remotefiles treeprocs/treeprocessormt_remotefiles.cxx LIBRARIES TreePlayer)
   endif()
endif()

ROOT_ADD_GTEST(ttreeindex_clone ttreeindex_clone.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(ttreereader_friends ttreereader_friends.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(ttreeindex_getlistoffriends ttreeindex_getlistoffriends.cxx LIBRARIES TreePlayer)

ROOT_ADD_GTEST(gh16804 gh16804.cxx LIBRARIES TreePlayer)
