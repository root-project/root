############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

set(libname HistPainter)

ROOT_GENERATE_DICTIONARY(G__${libname} *.h  MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")

if(root7)
    set(root7src v7/src/)
endif()

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx ${root7src} DEPENDENCIES  Graf Hist Matrix MathCore Gpad )
ROOT_INSTALL_HEADERS()

