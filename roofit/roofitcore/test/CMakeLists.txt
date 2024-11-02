# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

# @author Danilo Piparo CERN, 2018
# @author Patrick Bos, NL eScience Center, 2018

ROOT_ADD_GTEST(testSimple testSimple.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooAddPdf testRooAddPdf.cxx LIBRARIES RooFitCore RooStats)
ROOT_ADD_GTEST(testRooCacheManager testRooCacheManager.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooCategory testRooCategory.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testWorkspace testWorkspace.cxx LIBRARIES RooFitCore RooStats)
ROOT_ADD_GTEST(testRooDataHist testRooDataHist.cxx LIBRARIES RooFitCore
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/dataHistv4_ref.root
                   ${CMAKE_CURRENT_SOURCE_DIR}/dataHistv5_ref.root
                   ${CMAKE_CURRENT_SOURCE_DIR}/dataHistv6_ref.root)
ROOT_ADD_GTEST(testRooBinSamplingPdf testRooBinSamplingPdf.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooWrapperPdf testRooWrapperPdf.cxx LIBRARIES Gpad RooFitCore)
ROOT_ADD_GTEST(testGenericPdf testGenericPdf.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooAbsPdf testRooAbsPdf.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooAbsCollection testRooAbsCollection.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooDataSet testRooDataSet.cxx LIBRARIES Tree RooFitCore
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/dataSet_with_errors_6_26_10.root)
ROOT_ADD_GTEST(testRooFormula testRooFormula.cxx LIBRARIES RooFitCore ROOT::TestSupport)
ROOT_ADD_GTEST(testRooProdPdf testRooProdPdf.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testProxiesAndCategories testProxiesAndCategories.cxx
  LIBRARIES RooFitCore
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/testProxiesAndCategories_1.root
                   ${CMAKE_CURRENT_SOURCE_DIR}/categories_v620.root
                   ${CMAKE_CURRENT_SOURCE_DIR}/categories_v621.root
                   ${CMAKE_CURRENT_SOURCE_DIR}/categories_v622.root
                   ${CMAKE_CURRENT_SOURCE_DIR}/categories_v624.root)
ROOT_ADD_GTEST(testRooCollectionProxy testRooCollectionProxy.cxx
  LIBRARIES RooFitCore
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/testRooCollectionProxy_v626.root)
ROOT_ADD_GTEST(testRooRealVar testRooRealVar.cxx LIBRARIES RooFitCore
  COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/testRooRealVar_data1.root)

  ROOT_ADD_GTEST(testRooAbsReal testRooAbsReal.cxx
    LIBRARIES RooFitCore
    COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/testRooAbsReal_1.root ${CMAKE_CURRENT_SOURCE_DIR}/testRooAbsReal_2.root)
if(NOT MSVC OR win_broken_tests)
  # Disabled on Windows because it causes the following error:
  # unknown file: error: SEH exception with code 0xc0000005 thrown in the test body.
  # According to the internet, this has to do with gtest, so it's not a RooFit problem
  ROOT_ADD_GTEST(testRooRealIntegral testRooRealIntegral.cxx LIBRARIES RooFitCore)
endif()
if(clad)
  if(NOT MSVC OR MSVC_VERSION GREATER_EQUAL 1938)
    # Disabled on Windows with Visual Studio before v17.8 because it causes the following error:
    # Assertion failed: Ctx->isFileContext() && "We should have been looking
    # only at file context here already.", file
    # C:\build\workspace\root-pullrequests-build\root\interpreter\llvm-project\clang\lib\Sema\SemaLookup.cpp,
    # line 1492
    ROOT_ADD_GTEST(testRooFuncWrapper testRooFuncWrapper.cxx LIBRARIES RooFitCore RooFit HistFactory)
  endif()
endif()
ROOT_ADD_GTEST(testGlobalObservables testGlobalObservables.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testInterface TestStatistics/testInterface.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testLikelihoodSerial TestStatistics/testLikelihoodSerial.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testNaNPacker testNaNPacker.cxx LIBRARIES RooFitCore RooBatchCompute)
ROOT_ADD_GTEST(testRooAbsL TestStatistics/testRooAbsL.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooCurve testRooCurve.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooHist testRooHist.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooHistPdf testRooHistPdf.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooPolyFunc testRooPolyFunc.cxx LIBRARIES Gpad RooFitCore)
ROOT_ADD_GTEST(testRooRealL TestStatistics/testRooRealL.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooRombergIntegrator testRooRombergIntegrator.cxx LIBRARIES MathCore RooFitCore)
ROOT_ADD_GTEST(testRooSTLRefCountList testRooSTLRefCountList.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooSimultaneous testRooSimultaneous.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooTruthModel testRooTruthModel.cxx LIBRARIES RooFitCore RooFit)
ROOT_ADD_GTEST(testSumW2Error testSumW2Error.cxx LIBRARIES Gpad RooFitCore)
ROOT_ADD_GTEST(testTestStatistics testTestStatistics.cxx LIBRARIES RooFitCore)
ROOT_ADD_GTEST(testRooMinimizer testRooMinimizer.cxx LIBRARIES RooFitCore RooFit)
if (roofit_multiprocess)
  ROOT_ADD_GTEST(testTestStatisticsPlot TestStatistics/testPlot.cxx LIBRARIES RooFitMultiProcess RooFitCore RooFit
                   COPY_TO_BUILDDIR ${CMAKE_CURRENT_SOURCE_DIR}/TestStatistics/TestStatistics_ref.root)
  ROOT_ADD_GTEST(testLikelihoodGradientJob TestStatistics/testLikelihoodGradientJob.cxx LIBRARIES RooFitMultiProcess RooFitCore m ROOT::TestSupport)
  target_include_directories(testLikelihoodGradientJob PRIVATE ${RooFitCore_MultiProcess_TestStatistics_INCLUDE_DIR})
  ROOT_ADD_GTEST(testLikelihoodJob TestStatistics/testLikelihoodJob.cxx LIBRARIES RooFitMultiProcess RooFitCore m)
  target_include_directories(testLikelihoodJob PRIVATE ${RooFitCore_MultiProcess_TestStatistics_INCLUDE_DIR})
endif()

if(mathmore)
  list(APPEND ROOFITMORE_LIBRARIES RooFitMore)
endif()

#--stressRooFit----------------------------------------------------------------------------------
ROOT_EXECUTABLE(stressRooFit stressRooFit.cxx LIBRARIES RooFit Gpad ${ROOFITMORE_LIBRARIES})
if(mathmore)
    target_compile_definitions(stressRooFit PRIVATE ROOFITMORE)
endif()

configure_file(stressRooFit_ref.root stressRooFit_ref.root COPYONLY)
if(roofit_legacy_eval_backend)
  ROOT_ADD_TEST(test-stressroofit-legacy COMMAND stressRooFit -b legacy FAILREGEX "FAILED|Error in")
endif()
ROOT_ADD_TEST(test-stressroofit-cpu COMMAND stressRooFit -b cpu FAILREGEX "FAILED|Error in")
if(cuda)
  ROOT_ADD_TEST(test-stressroofit-cuda COMMAND stressRooFit -b cuda FAILREGEX "FAILED|Error in" RESOURCE_LOCK GPU)
endif()
if(dataframe)
  ROOT_ADD_GTEST(testActionHelpers testActionHelpers.cxx LIBRARIES ROOTDataFrame RooFitCore)
endif()
