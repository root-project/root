############################################################################
# CMakeLists.txt file for building ROOT core/textinout package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
                          ${CMAKE_CURRENT_SOURCE_DIR}/src/textinput/*.cpp)

ROOT_GLOB_HEADERS(headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
set(TextInput_dict_headers ${headers} PARENT_SCOPE)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
ROOT_OBJECT_LIBRARY(TextInput ${sources})

ROOT_INSTALL_HEADERS()
set_property(GLOBAL APPEND PROPERTY ROOT_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)
