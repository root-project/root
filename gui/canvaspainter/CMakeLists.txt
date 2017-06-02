############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

set(libname ROOTCanvasPainter)

ROOT_LINKER_LIBRARY(${libname} *.cxx v7/src/ DEPENDENCIES Gpad RHTTP)
