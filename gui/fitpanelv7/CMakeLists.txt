############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFitPanelv7
                              HEADERS inc/ROOT/*.hxx
                              SOURCES src/*.cxx
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES ROOTWebDisplay ROOTGpadv7 ROOTHistDraw)
