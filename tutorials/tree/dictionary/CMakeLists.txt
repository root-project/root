#####################################################################################################################

# Jorge Agramunt Ros    @ IFIC(Valencia,Spain)  jorge.agramunt@ific.uv.es
# Alvaro Tolosa Delgado @ IFIC(Valencia,Spain)  alvaro.tolosa@ific.uv.es
# Copyright (c) 2016 Jorge Agramunt & Alvaro Tolosa. All rights reserved.

#####################################################################################################################

# Details about integrating ROOT into CMake projects:
#     https://root.cern/manual/integrate_root_into_my_cmake_project/

#####################################################################################################################

# CMakeLists.txt for event package. It creates a library with dictionary and a main program
cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

project(treeUsingCustomClass)

#---Locate the ROOT package and defines a number of variables (e.g. ROOT_INCLUDE_DIRS)
find_package(ROOT REQUIRED COMPONENTS MathCore RIO Hist Tree Net Graf3d Graf Gpad Thread TreePlayer)

#---Define useful ROOT functions and macros (e.g. ROOT_GENERATE_DICTIONARY)
include(${ROOT_USE_FILE})

#---Add include directory of ROOT to the build
include_directories(${CMAKE_SOURCE_DIR})

# CMake function provided by ROOT, used to generate the dictionary file, G__data2Tree.cxx
#     See this link for further details:
#     https://root.cern/manual/io_custom_classes/#using-cmake
ROOT_GENERATE_DICTIONARY(G__data2Tree data2Tree.hpp LINKDEF data2TreeLinkDef.hpp)

#---Create a shared library from
#   * the previously generated dictionary, G__data2Tree.cxx
#   * the class implementation
add_library(data2TreeLib SHARED data2Tree.cpp G__data2Tree.cxx)
target_link_libraries(data2TreeLib ${ROOT_LIBRARIES}  ) 

#---Create  a main program using the library
add_executable(treeExample main.cpp readTree.cpp writeTree.cpp)
target_link_libraries(treeExample ${ROOT_LIBRARIES} data2TreeLib)

