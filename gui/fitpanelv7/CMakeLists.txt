############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFitPanelv7
  HEADERS
    ROOT/RFitPanel.hxx
    ROOT/RFitPanel6.hxx
  SOURCES  
    RFitPanel.cxx
    RFitPanel6.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    Gpad
    ROOTGpadv7
    ROOTHistDraw
    ROOTWebDisplay
)
