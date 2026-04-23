# Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN

ROOT_EXECUTABLE(fitPanelUnitTesting UnitTesting.cxx LIBRARIES FitPanel Gui Tree Hist MathCore)

# batch usage is instable - need more investigations
#ROOT_ADD_TEST(test-fitpanel-UnitTesting-batch COMMAND fitPanelUnitTesting -b FAILREGEX "FAILED|Error in")

find_program(XVFB_RUN_EXECUTABLE NAMES xvfb-run)

if(XVFB_RUN_EXECUTABLE)
   ROOT_ADD_TEST(test-fitpanel-UnitTesting-xvfb
                 COMMAND ${XVFB_RUN_EXECUTABLE} --auto-servernum --server-args="-screen 0 1024x768x24" ./fitPanelUnitTesting
                 PASSRC 1
                 FAILREGEX "FAILED|Error in")
endif()
