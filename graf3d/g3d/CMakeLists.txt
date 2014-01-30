############################################################################
# CMakeLists.txt file for building ROOT graf3d/g3d package
############################################################################

set(libname Graf3d)

ROOT_USE_PACKAGE(hist/hist)

ROOT_GLOB_HEADERS(headers1 ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers1 ${CMAKE_CURRENT_SOURCE_DIR}/inc/X3DBuffer.h)
set(headers2 X3DBuffer.h )

ROOT_GENERATE_DICTIONARY(G__${libname} ${headers1} MODULE ${libname} LINKDEF LinkDef.h)

ROOT_LINKER_LIBRARY(${libname} *.cxx *.c G__${libname}.cxx DEPENDENCIES Graf Hist Gpad MathCore)
ROOT_INSTALL_HEADERS()

