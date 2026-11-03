# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT gui/fitpanelv7
############################################################################

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

set(FITPANELV7_HEADERS
  ROOT/RFitPanel.hxx
  ROOT/RFitPanelModel.hxx
)

ROOT_STANDARD_LIBRARY_PACKAGE(ROOTFitPanelv7
  HEADERS
    ${FITPANELV7_HEADERS}
  SOURCES
    RFitPanel.cxx
    RFitPanelModel.cxx
  DICTIONARY_OPTIONS
    -writeEmptyRootPCM
  DEPENDENCIES
    Hist
    Gpad
    ROOTWebDisplay
    ROOTGpadv7
  ${EXTRA_DICT_OPTS}
)

set(RELATIVE_INC_HEADERS ${FITPANELV7_HEADERS})
list(TRANSFORM RELATIVE_INC_HEADERS PREPEND inc/)
target_sources(ROOTFitPanelv7
PRIVATE
  FILE_SET private_header_files
  TYPE HEADERS
  BASE_DIRS inc/
  FILES
    ${RELATIVE_INC_HEADERS}
    inc/LinkDef.h
)
