ROOT_GLOB_SOURCES(Eve_v7_sources      ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx)
ROOT_GLOB_HEADERS(Eve_v7_dict_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTEve
    HEADERS   ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx
    SOURCES   ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
    LIBRARIES 
    DEPENDENCIES Core Geom Physics EG TreePlayer ROOTWebDisplay
    LINKDEF   ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h)

include_directories(".")
set_source_files_properties(src/TEveGluTess.cxx PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
