############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

ROOT_GLOB_HEADERS(FitPanelv7_headers inc/ROOT/T*.hxx)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFitPanelv7
                              HEADERS ${FitPanelv7_headers}
                              DICTIONARY_OPTIONS "-writeEmptyRootPCM"
                              DEPENDENCIES ROOTWebDisplay ROOTGpadv7 ROOTHistDraw)
