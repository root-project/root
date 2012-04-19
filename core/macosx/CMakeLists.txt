############################################################################
# CMakeLists.txt file for building ROOT core/macosx package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm)

ROOT_GENERATE_DICTIONARY(G__Macosx T*.h LINKDEF LinkDef.h)
set(MACOSX_DICTIONARY ${CMAKE_CURRENT_BINARY_DIR}/G__Macosx.cxx  PARENT_SCOPE)

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h PARENT_SCOPE)
add_custom_target(MACOSX_DICTIONARY DEPENDS G__Macosx.cxx)
set_target_properties(MACOSX_DICTIONARY PROPERTIES FOLDER Dictionaries)

ROOT_INSTALL_HEADERS()
