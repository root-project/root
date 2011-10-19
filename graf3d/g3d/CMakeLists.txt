############################################################################
# CMakeLists.txt file for building ROOT graf3d/g3d package
############################################################################

set(libname Graf3d)

ROOT_USE_PACKAGE(hist/hist)

file(GLOB headers1 ${CMAKE_CURRENT_SOURCE_DIR}/inc/*.h)
list(REMOVE_ITEM headers1 ${CMAKE_CURRENT_SOURCE_DIR}/inc/LinkDef.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/inc/X3DBuffer.h
                          ${CMAKE_CURRENT_SOURCE_DIR}/inc/X3DDefs.h)
set(headers2 X3DBuffer.h X3DDefs.h)

ROOT_GENERATE_DICTIONARY(G__G3D ${headers1} LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES Graf Hist Gpad MathCore)
ROOT_LINKER_LIBRARY(${libname} *.cxx *.c G__G3D.cxx LIBRARIES Graf Hist Gpad MathCore)
ROOT_INSTALL_HEADERS()

