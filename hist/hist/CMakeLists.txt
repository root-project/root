############################################################################
# CMakeLists.txt file for building ROOT hist/hist package
############################################################################

set(libname Hist)

include_directories(${CMAKE_SOURCE_DIR}/graf3d/g3d/inc)   # This is to avoid a circular dependency g3d <--> hist

ROOT_GENERATE_DICTIONARY(G__${libname} *.h Math/*.h v5/*.h MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

if(root7)
    set(root7src v7/src/)
endif()

ROOT_LINKER_LIBRARY(${libname} *.cxx ${root7src} G__${libname}.cxx DEPENDENCIES Matrix MathCore)
ROOT_INSTALL_HEADERS()

