############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_GLOB_HEADERS(WebDisplay_headers inc/ROOT/R*.hxx)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
                              HEADERS ${WebDisplay_headers}
                              DEPENDENCIES RHTTP MathCore)
