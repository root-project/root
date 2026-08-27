# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Stephan Hageboeck CERN, 2019

if(NOT MSVC OR win_broken_tests)
  ROOT_ADD_GTEST(stressHistFactory stressHistFactory.cxx
    LIBRARIES RooFitCore RooFit RooStats HistFactory
    COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/HistFactoryTest.tar ${CMAKE_CURRENT_SOURCE_DIR}/HistFactoryTest.tar)
endif()

ROOT_ADD_GTEST(testHistFactory testHistFactory.cxx
  LIBRARIES RooFitCore RooFit RooStats HistFactory RooFitHS3
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/ref_6.16_example_UsingC_channel1_meas_model.root ${CMAKE_CURRENT_SOURCE_DIR}/ref_6.16_example_UsingC_combined_meas_model.root)

if(clad)
  target_compile_definitions(testHistFactory PUBLIC TEST_CODEGEN_AD)
endif(clad)

ROOT_ADD_GTEST(testParamHistFunc testParamHistFunc.cxx LIBRARIES RooFitCore HistFactory)
ROOT_ADD_GTEST(testPiecewiseInterpolation testPiecewiseInterpolation.cxx LIBRARIES RooFitCore HistFactory)
ROOT_ADD_GTEST(testHistFactoryPlotting testHistFactoryPlotting.cxx LIBRARIES RooFitCore HistFactory)
