############################################################################
# CMakeLists.txt file for building ROOT hist/spectrumpainter package
############################################################################

set(libname SpectrumPainter)

ROOT_GENERATE_DICTIONARY(G__SpectrumPainter *.h MODULE ${libname} LINKDEF LinkDef.h OPTIONS "-writeEmptyRootPCM" DEPENDENCIES Graf Hist)


ROOT_LINKER_LIBRARY(${libname} *.cxx G__SpectrumPainter.cxx DEPENDENCIES Graf Hist )
ROOT_INSTALL_HEADERS()

