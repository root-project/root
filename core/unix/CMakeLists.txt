############################################################################
# CMakeLists.txt file for building ROOT core/unix package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__Unix *.h LINKDEF LinkDef.h)
set(UNIX_DICTIONARY ${CMAKE_CURRENT_BINARY_DIR}/G__Unix.cxx  PARENT_SCOPE)

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h PARENT_SCOPE)
add_custom_target(UNIX_DICTIONARY DEPENDS G__Unix.cxx)
set_target_properties(UNIX_DICTIONARY PROPERTIES FOLDER Dictionaries)

ROOT_INSTALL_HEADERS()
