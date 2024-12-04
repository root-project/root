# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

ROOT_ADD_GTEST(testTProfile2Poly test_tprofile2poly.cxx LIBRARIES Hist Matrix MathCore RIO)
ROOT_ADD_GTEST(testTH2PolyBinError test_TH2Poly_BinError.cxx LIBRARIES Hist Matrix MathCore RIO)
ROOT_ADD_GTEST(testTH2PolyAdd test_TH2Poly_Add.cxx LIBRARIES Hist Matrix MathCore RIO)
ROOT_ADD_GTEST(testTH2PolyGetNumberOfBins test_TH2Poly_GetNumberOfBins.cxx LIBRARIES Hist Matrix MathCore RIO)
ROOT_ADD_GTEST(testTHn THn.cxx LIBRARIES Hist Matrix MathCore RIO)
ROOT_ADD_GTEST(testTH1 test_TH1.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(testTHStack test_THStack.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(testProject3Dname test_Project3D_name.cxx LIBRARIES Hist)
if(geom)
  ROOT_ADD_GTEST(testTFormula test_TFormula.cxx LIBRARIES Hist)
endif()
ROOT_ADD_GTEST(testTKDE test_tkde.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(testTH1FindFirstBinAbove test_TH1_FindFirstBinAbove.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(test_TEfficiency test_TEfficiency.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(TGraphMultiErrorsTests TGraphMultiErrorsTests.cxx LIBRARIES Hist RIO)
ROOT_ADD_GTEST(test_TF123_Moments test_TF123_Moments.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(test_THBinIterator test_THBinIterator.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(testTMultiGraphGetHistogram test_TMultiGraph_GetHistogram.cxx LIBRARIES Hist Gpad)
ROOT_ADD_GTEST(testMapCppName test_MapCppName.cxx LIBRARIES Hist Gpad)
ROOT_ADD_GTEST(testTGraphSorting test_TGraph_sorting.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(testSpline test_spline.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(testTF1Simple test_tf1_simple.cxx LIBRARIES Hist RIO)

if(fftw3)
  ROOT_ADD_GTEST(testTF1 test_tf1.cxx LIBRARIES Hist)
endif()

ROOT_ADD_GTEST(testTF2 test_tf2.cxx LIBRARIES Hist)
ROOT_ADD_GTEST(testTF3 test_tf3.cxx LIBRARIES Hist)

ROOT_ADD_GTEST(testTH1sa test_TH1_SaveAs.cxx LIBRARIES Hist)

if(clad)
  ROOT_ADD_GTEST(TFormulaGradientTests TFormulaGradientTests.cxx LIBRARIES Core MathCore Hist)
  ROOT_ADD_GTEST(TFormulaHessianTests TFormulaHessianTests.cxx LIBRARIES Core MathCore Hist)
endif()

# Regression test for https://github.com/root-project/root/issues/13691
# The test needs the following to be actually useful:
# - It must be compiled and run as a standalone executable, as it depends on
#   gROOT not being initialized yet.
# - It must link to any library that is not in `core`. See the linked issue
#   for details.
ROOT_EXECUTABLE(tdirectoryfile_destructor_segfault tdirectoryfile_destructor_segfault.cxx LIBRARIES RIO Hist)
ROOT_ADD_TEST(test-tdirectoryfile_destructor_segfault COMMAND tdirectoryfile_destructor_segfault)

# Test for https://github.com/root-project/root/issues/6658
ROOT_ADD_GTEST(test_projections test_projections.cxx LIBRARIES Hist)
