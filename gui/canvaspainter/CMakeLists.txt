############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_LINKER_LIBRARY(ROOTCanvasPainter
  v7/src/TCanvasPainter.cxx
  DEPENDENCIES
    ROOTGpadv7
    RHTTP
    ROOTWebDisplay
)
