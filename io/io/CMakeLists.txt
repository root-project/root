############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

set(libname RIO)

include_directories(${CMAKE_SOURCE_DIR}/core/clib/res)

ROOT_GENERATE_DICTIONARY(G__RIO *.h ROOT/*.hxx STAGE1 MODULE ${libname} LINKDEF LinkDef.h DEPENDENCIES Core Thread)

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
endif()

ROOT_OBJECT_LIBRARY(RIOObjs G__RIO.cxx  ${root7src} *.cxx)
ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:RIOObjs> $<TARGET_OBJECTS:RootPcmObjs>
                               LIBRARIES ${CMAKE_DL_LIBS}
                               DEPENDENCIES Core Thread)
ROOT_INSTALL_HEADERS()

if(testing)
  add_subdirectory(test)
endif()

#--- Extra rules ----------------------------------------------------------
