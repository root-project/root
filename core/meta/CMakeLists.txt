############################################################################
# CMakeLists.txt file for building ROOT core/meta package
############################################################################

ROOT_GLOB_SOURCES(sources ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
ROOT_GLOB_HEADERS(headers RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/inc ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)

if(cling)
  add_definitions(-DR__WITH_CLING)
else()
  list(REMOVE_ITEM headers TCintWithCling.h)
  list(REMOVE_ITEM sources ${CMAKE_CURRENT_SOURCE_DIR}/src/TCintWithCling.cxx)
endif()

ROOT_GENERATE_DICTIONARY(G__Meta ${headers} LINKDEF LinkDef.h OPTIONS -DG__API)
set(dictionaries ${CMAKE_CURRENT_BINARY_DIR}/G__Meta.cxx)

set(LibCore_SRCS ${LibCore_SRCS} ${sources} PARENT_SCOPE)
set(LibCore_LINKDEF ${LibCore_LINKDEF} ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h PARENT_SCOPE)

add_custom_target(META_DICTIONARY DEPENDS ${dictionaries})
set(META_DICTIONARY ${dictionaries}  PARENT_SCOPE)
set_target_properties(META_DICTIONARY PROPERTIES FOLDER Dictionaries)

ROOT_INSTALL_HEADERS()
