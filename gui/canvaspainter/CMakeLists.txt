############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_GLOB_SOURCES(sources v7/src/*.cxx)

ROOT_LINKER_LIBRARY(ROOTCanvasPainter ${sources} DEPENDENCIES ROOTGpadv7 RHTTP ROOTWebDisplay)
