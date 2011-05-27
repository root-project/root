############################################################################
# CMakeLists.txt file for building ROOT hist/histpainter package
############################################################################

set(libname HistPainter)

ROOT_USE_PACKAGE(hist/hist)


ROOT_GENERATE_DICTIONARY(G__${libname} *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES  Graf Hist Matrix MathCore Gpad )

ROOT_LINKER_LIBRARY(${libname} *.cxx G__${libname}.cxx LIBRARIES  Graf Hist Matrix MathCore Gpad )
ROOT_INSTALL_HEADERS()

