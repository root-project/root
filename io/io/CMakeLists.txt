############################################################################
# CMakeLists.txt file for building ROOT io/io package
############################################################################

set(libname RIO)

ROOT_GENERATE_DICTIONARY(G__IO *.h STAGE1 MODULE ${libname} LINKDEF LinkDef.h)

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
    message("AXEL: " ${root7src})
endif()

ROOT_OBJECT_LIBRARY(RIOObjs G__IO.cxx  ${root7src} *.cxx)
ROOT_LINKER_LIBRARY(${libname} $<TARGET_OBJECTS:RIOObjs>
                               LIBRARIES ${CMAKE_DL_LIBS}
                               DEPENDENCIES Core Thread)
ROOT_INSTALL_HEADERS()

#--- Extra rules ----------------------------------------------------------
