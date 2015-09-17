############################################################################
# CMakeLists.txt file for building ROOT hist/hist package
############################################################################

set(libname Hist)

include_directories(${CMAKE_SOURCE_DIR}/graf3d/g3d/inc)   # This is to avoid a circular dependency g3d <--> hist

ROOT_GENERATE_DICTIONARY(G__${libname} *.h Math/*.h v5/*.h MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
endif()

ROOT_LINKER_LIBRARY(${libname} *.cxx ${root7src} G__${libname}.cxx DEPENDENCIES Matrix MathCore)
ROOT_INSTALL_HEADERS()

