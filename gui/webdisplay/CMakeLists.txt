############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_GLOB_HEADERS(WebDisplay_headers ${CMAKE_CURRENT_SOURCE_DIR}/inc/ROOT/T*.hxx)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
                              HEADERS ${WebDisplay_headers}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES RHTTP MathCore)
