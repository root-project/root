# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanel package
############################################################################

set(FITPANEL_HEADERS
  TAdvancedGraphicsDialog.h
  TFitEditor.h
  TFitParametersDialog.h
  TTreeInput.h
)

ROOT_STANDARD_LIBRARY_PACKAGE(FitPanel
  HEADERS
    ${FITPANEL_HEADERS}
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
    TreePlayer
)

set(RELATIVE_INC_HEADERS ${FITPANEL_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(FitPanel
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/ src/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
    src/CommonDefs.h
)

ROOT_ADD_TEST_SUBDIRECTORY(test)
