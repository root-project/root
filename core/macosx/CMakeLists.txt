############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm)
ROOT_GLOB_SOURCES(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/T*.h)

set(Macosx_dict_headers ${headers} PARENT_SCOPE)

ROOT_OBJECT_LIBRARY(Macosx ${sources})
set_source_files_properties(${sources} COMPILE_FLAGS "-ObjC++ -std=c++11")

ROOT_INSTALL_HEADERS()
