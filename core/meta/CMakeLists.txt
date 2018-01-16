############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

ROOT_GLOB_FILES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
ROOT_GLOB_FILES(headers FILTER "(LinkDef|RootMetaSelection.h)" ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)

set(Meta_dict_headers ${headers} PARENT_SCOPE)
ROOT_OBJECT_LIBRARY(Meta ${sources})

ROOT_INSTALL_HEADERS()

ROOT_ADD_TEST_SUBDIRECTORY(test)
