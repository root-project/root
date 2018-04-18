############################################################################
# CMakeLists.txt file for building ROOT gui/canvaspainter package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTWebDisplay
                              HEADERS inc/ROOT/*.hxx
                              SOURCES src/*.cxx
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES RHTTP MathCore)
