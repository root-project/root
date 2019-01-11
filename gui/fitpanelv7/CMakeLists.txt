############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFitPanelv7
  HEADERS
    ROOT/RFitPanel.hxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    ROOTGpadv7
    ROOTHistDraw
    ROOTWebDisplay
)
