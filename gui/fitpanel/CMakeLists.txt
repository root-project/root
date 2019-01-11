############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanel package
############################################################################

ROOT_STANDARD_LIBRARY_PACKAGE(FitPanel
  HEADERS
    TAdvancedGraphicsDialog.h
    TFitEditor.h
    TFitParametersDialog.h
    TTreeInput.h
  SOURCES
    src/TAdvancedGraphicsDialog.cxx
    src/TFitEditor.cxx
    src/TFitParametersDialog.cxx
    src/TTreeInput.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Gpad
    Graf
    Gui
    Hist
    MathCore
    Tree
)
