# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

ROOT_EXECUTABLE(fitPanelUnitTesting UnitTesting.cxx LIBRARIES FitPanel Gui Tree Hist MathCore)
ROOT_ADD_TEST(test-fitpanel-UnitTesting COMMAND fitPanelUnitTesting -b FAILREGEX "FAILED|Error in")
target_include_directories(fitPanelUnitTesting PRIVATE ../src/)
