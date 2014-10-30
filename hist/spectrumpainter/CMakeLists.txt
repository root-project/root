############################################################################
# CMakeLists.txt file for building ROOT hist/spectrumpainter package
############################################################################

set(libname SpectrumPainter)

ROOT_USE_PACKAGE(hist/hist)


ROOT_GENERATE_DICTIONARY(G__Spectrum2Painter *.h MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM")


ROOT_LINKER_LIBRARY(${libname} *.cxx G__Spectrum2Painter.cxx DEPENDENCIES Graf Hist )
ROOT_INSTALL_HEADERS()

