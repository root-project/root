############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFitPanelv7
  HEADERS
    ROOT/RFitPanel.hxx
    ROOT/RFitPanelModel.hxx
  SOURCES
    RFitPanel.cxx
    RFitPanelModel.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    Gpad
    ROOTGpadv7
    ROOTHistDraw
    ROOTWebDisplay
)
