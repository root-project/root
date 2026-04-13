# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

ROOT_EXECUTABLE(fitPanelUnitTesting UnitTesting.cxx LIBRARIES FitPanel Gui Tree Hist MathCore)
ROOT_ADD_TEST(test-fitpanel-UnitTesting COMMAND fitPanelUnitTesting)
# set_tests_properties(test-fitpanel-UnitTesting PROPERTIES WILL_FAIL true) # Problems with batch mode https://github.com/root-project/root/issues/21881
target_include_directories(fitPanelUnitTesting PRIVATE ../src/)
