ROOT_STANDARD_LIBRARY_PACKAGE(ROOTEve
    HEADERS   ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/*.hxx
    SOURCES   ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cxx
    LIBRARIES 
    DEPENDENCIES Core Geom Physics EG TreePlayer ROOTWebDisplay
    LINKDEF   ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h)

include_directories(${CMAKE_SOURCE_DIR}/io/io/res ${CMAKE_SOURCE_DIR}/graf3d/eve7)

# set_source_files_properties(src/REveGluTess.cxx PROPERTIES COMPILE_FLAGS -Wno-unused-parameter)
