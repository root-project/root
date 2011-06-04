############################################################################
# CMakeLists.txt file for building ROOT core/clib package
############################################################################

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c)

ROOT_GENERATE_DICTIONARY(G__Clib strlcpy.h snprintf.h LINKDEF LinkDef.h)

set(CLIB_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/G__Clib.cxx PARENT_SCOPE)

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h PARENT_SCOPE)
add_custom_target(CLIB_DICTIONARY DEPENDS G__Clib.cxx)
set_target_properties(CLIB_DICTIONARY PROPERTIES FOLDER Dictionaries)
ROOT_INSTALL_HEADERS()




