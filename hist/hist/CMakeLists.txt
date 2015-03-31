############################################################################
# CMakeLists.txt file for building ROOT hist/hist package
############################################################################

set(libname Hist)

ROOT_USE_PACKAGE(core)
ROOT_USE_PACKAGE(math)
ROOT_USE_PACKAGE(graf2d)
ROOT_USE_PACKAGE(io/io)
include_directories(${CMAKE_SOURCE_DIR}/graf3d/g3d/inc)   # This is to avoid a circular dependency g3d <--> hist

ROOT_GENERATE_DICTIONARY(G__${libname} *.h Math/*.h v5/*.h MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES Matrix MathCore)
ROOT_INSTALL_HEADERS()

