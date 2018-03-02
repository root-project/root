############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

ROOT_ADD_TEST_SUBDIRECTORY(test)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/res ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/res)

ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx)
set(Cont_dict_headers ${headers} PARENT_SCOPE)

ROOT_OBJECT_LIBRARY(Cont *.cxx)

ROOT_INSTALL_HEADERS()
