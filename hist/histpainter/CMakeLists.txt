############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

set(libname HistPainter)

ROOT_GENERATE_DICTIONARY(G__${libname} *.h  MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES  Graf Hist Matrix MathCore Gpad)

if(root7)
    ROOT_GLOB_SOURCES(root7src RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} v7/src/*.cxx)
endif()

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx ${root7src} DEPENDENCIES  Graf Hist Matrix MathCore Gpad )
ROOT_INSTALL_HEADERS()

