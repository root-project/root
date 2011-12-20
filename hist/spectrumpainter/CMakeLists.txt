############################################################################
# CMakeLists.txt file for building ROOT hist/spectrumpainter package
############################################################################

set(libname SpectrumPainter)

ROOT_USE_PACKAGE(hist/hist)


ROOT_GENERATE_DICTIONARY(G__Spectrum2Painter *.h  LINKDEF LinkDef.h)
ROOT_GENERATE_ROOTMAP(${libname} LINKDEF LinkDef.h DEPENDENCIES  Graf Hist )

ROOT_LINKER_LIBRARY(${libname} *.cxx G__Spectrum2Painter.cxx DEPENDENCIES Graf Hist )
ROOT_INSTALL_HEADERS()

