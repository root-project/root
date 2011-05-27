############################################################################
# CMakeLists.txt file for building ROOT core/cont package
############################################################################

file(GLOB sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)

ROOT_GENERATE_DICTIONARY(G__Cont *.h LINKDEF LinkDef.h)

set(CONT_DICTIONARY  ${CMAKE_CURRENT_BINARY_DIR}/G__Cont.cxx PARENT_SCOPE)

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h PARENT_SCOPE)
add_custom_target(CONT_DICTIONARY DEPENDS G__Cont.cxx)
set_target_properties(CONT_DICTIONARY PROPERTIES FOLDER Dictionaries)
ROOT_INSTALL_HEADERS()
