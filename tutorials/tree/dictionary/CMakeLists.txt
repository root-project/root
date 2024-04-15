# Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

#####################################################################################################################

# Details about integrating ROOT into CMake projects:
#     https://root.cern/manual/integrate_root_into_my_cmake_project/

#####################################################################################################################

# CMakeLists.txt that creates a library with dictionary and a main program
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(treeUsingCustomClass)

#---Locate the ROOT package and defines a number of variables (e.g. ROOT_INCLUDE_DIRS)
find_package(ROOT REQUIRED COMPONENTS Tree TreePlayer ROOTDataFrame)

#---Include a CMake module which makes use of the previous variables and loads modules 
#   with useful macros or functions such as ROOT_GENERATE_DICTIONARY
#   For further details: https://root-forum.cern.ch/t/how-to-integrate-root-into-my-project-with-cmake/37175
include(${ROOT_USE_FILE})

#---Add include directory of ROOT to the build
include_directories(${CMAKE_SOURCE_DIR})

# CMake function provided by ROOT, used to generate the dictionary file, G__data2Tree.cxx
#     See this link for further details:
#     https://root.cern/manual/io_custom_classes/#using-cmake
ROOT_GENERATE_DICTIONARY(G__data2Tree data2Tree.hxx LINKDEF data2TreeLinkDef.hxx)

#---Create a shared library from
#   * the previously generated dictionary, G__data2Tree.cxx
#   * the class implementation
add_library(data2TreeLib SHARED data2Tree.cxx G__data2Tree.cxx)
target_link_libraries(data2TreeLib ${ROOT_LIBRARIES}  ) 
add_dependencies(data2TreeLib G__data2Tree  )

#--- This is needed on Windows in order to export the symbols and create the data2TreeLib.lib file
if(MSVC)
  set_target_properties(data2TreeLib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
endif()

#---Create  a main program using the library
add_executable(treeExample main.cpp writeTree.cxx readTree.cxx readTreeDF.cxx)
target_link_libraries(treeExample ${ROOT_LIBRARIES} data2TreeLib)

