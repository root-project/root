############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

set(libname HistPainter)

ROOT_USE_PACKAGE(hist/hist)


ROOT_GENERATE_DICTIONARY(G__${libname} *.h  MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")


ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx DEPENDENCIES  Graf Hist Matrix MathCore Gpad )
ROOT_INSTALL_HEADERS()

