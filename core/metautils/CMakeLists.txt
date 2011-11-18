############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
file(GLOB headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

if(NOT cling)
  list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/TMetaUtils.cxx)
  list(REMOVE_ITEM headers TMetaUtils.h)
endif()

ROOT_GENERATE_DICTIONARY(G__MetaUtils ${headers} LINKDEF LinkDef.h)

set(METAUTILS_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/G__MetaUtils.cxx PARENT_SCOPE)

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h PARENT_SCOPE)
set(METAUTILS_CXX_SOURCES  ${sources} PARENT_SCOPE)
add_custom_target(METAUTILS_DICTIONARY DEPENDS G__MetaUtils.cxx)
set_target_properties(METAUTILS_DICTIONARY PROPERTIES FOLDER Dictionaries)

ROOT_INSTALL_HEADERS()
