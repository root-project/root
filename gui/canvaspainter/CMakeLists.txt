############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

set(libname ROOTCanvasPainter)

ROOT_GLOB_SOURCES(sources v7/src/*.cxx)

ROOT_LINKER_LIBRARY(${libname} ${sources} DEPENDENCIES Gpad RHTTP ROOTWebDisplay)
