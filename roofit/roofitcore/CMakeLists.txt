# Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.
# All rights reserved.
#
# For the licensing terms see $ROOTSYS/LICENSE.
# For the list of contributors see $ROOTSYS/README/CREDITS.

############################################################################
# CMakeLists.txt file for building ROOT roofitcore package
# @author Pere Mato, CERN
############################################################################

if(NOT WIN32)
  set(RooFitMPTestStatisticsSources
    src/TestStatistics/LikelihoodGradientJob.cxx
    src/TestStatistics/LikelihoodJob.cxx
    src/TestStatistics/MinuitFcnGrad.cxx
  )
  set(RooFitMPTestStatisticsHeaders
    src/TestStatistics/LikelihoodGradientJob.h
    src/TestStatistics/LikelihoodSerial.h
    src/TestStatistics/MinuitFcnGrad.h
    src/TestStatistics/LikelihoodJob.h
  )
  list(APPEND EXTRA_LIBRARIES RooFitMultiProcess)
  #FIXME: The ProcessTimer.h exposes json in its interface:
  list(APPEND EXTRA_LIBRARIES nlohmann_json::nlohmann_json)
  list(APPEND EXTRA_DEPENDENCIES Minuit2)
endif()

if(roofit_legacy_eval_backend)
  set(LegacyEvalBackendSources
    src/BidirMMapPipe.cxx
    src/RooAbsOptTestStatistic.cxx
    src/RooAbsTestStatistic.cxx
    src/RooChi2Var.cxx
    src/RooNLLVar.cxx
    src/RooRealMPFE.cxx
  )
  set(LegacyEvalBackendHeaders
    src/BidirMMapPipe.h
    src/RooAbsOptTestStatistic.h
    src/RooAbsTestStatistic.h
    src/RooChi2Var.h
    src/RooNLLVar.h
    src/RooRealMPFE.h
  )
endif()

set (EXTRA_DICT_OPTS)
if (runtime_cxxmodules AND WIN32)
  set (EXTRA_DICT_OPTS NO_CXXMODULE)
endif()

add_library(RooFitCore SHARED)
target_sources(RooFitCore
PRIVATE
  src/BatchModeDataHelpers.cxx
  src/ConstraintHelpers.cxx
  src/FitHelpers.cxx
  src/Initialisation.cxx
  src/ModelConfig.cxx
  src/NormalizationHelpers.cxx
  src/Roo1DTable.cxx
  src/RooAICRegistry.cxx
  src/RooAbsAnaConvPdf.cxx
  src/RooAbsArg.cxx
  src/RooAbsBinning.cxx
  src/RooAbsCache.cxx
  src/RooAbsCacheElement.cxx
  src/RooAbsCachedPdf.cxx
  src/RooAbsCachedReal.cxx
  src/RooAbsCategory.cxx
  src/RooAbsCategoryLValue.cxx
  src/RooAbsCollection.cxx
  src/RooAbsData.cxx
  src/RooAbsDataHelper.cxx
  src/RooAbsDataStore.cxx
  src/RooAbsFunc.cxx
  src/RooAbsGenContext.cxx
  src/RooAbsHiddenReal.cxx
  src/RooAbsIntegrator.cxx
  src/RooAbsLValue.cxx
  src/RooAbsMCStudyModule.cxx
  src/RooAbsMinimizerFcn.cxx
  src/RooAbsMoment.cxx
  src/RooAbsNumGenerator.cxx
  src/RooAbsPdf.cxx
  src/RooAbsProxy.cxx
  src/RooAbsReal.cxx
  src/RooAbsRealLValue.cxx
  src/RooAbsStudy.cxx
  src/RooAcceptReject.cxx
  src/RooAdaptiveIntegratorND.cxx
  src/RooAddGenContext.cxx
  src/RooAddHelpers.cxx
  src/RooAddModel.cxx
  src/RooAddPdf.cxx
  src/RooAddition.cxx
  src/RooArgList.cxx
  src/RooArgProxy.cxx
  src/RooArgSet.cxx
  src/RooBinIntegrator.cxx
  src/RooBinSamplingPdf.cxx
  src/RooBinWidthFunction.cxx
  src/RooBinnedGenContext.cxx
  src/RooBinning.cxx
  src/RooBinningCategory.cxx
  src/RooBrentRootFinder.cxx
  src/RooCachedPdf.cxx
  src/RooCachedReal.cxx
  src/RooCategory.cxx
  src/RooChangeTracker.cxx
  src/RooClassFactory.cxx
  src/RooCmdArg.cxx
  src/RooCmdConfig.cxx
  src/RooCompositeDataStore.cxx
  src/RooConstVar.cxx
  src/RooConstraintSum.cxx
  src/RooConvCoefVar.cxx
  src/RooConvGenContext.cxx
  src/RooConvIntegrandBinding.cxx
  src/RooCurve.cxx
  src/RooCustomizer.cxx
  src/RooDLLSignificanceMCSModule.cxx
  src/RooDataHist.cxx
  src/RooDataHistSliceIter.cxx
  src/RooDataProjBinding.cxx
  src/RooDataSet.cxx
  src/RooDerivative.cxx
  src/RooDirItem.cxx
  src/RooDouble.cxx
  src/RooEffGenContext.cxx
  src/RooEffProd.cxx
  src/RooEfficiency.cxx
  src/RooEllipse.cxx
  src/RooErrorVar.cxx
  src/RooEvaluatorWrapper.cxx
  src/RooExpensiveObjectCache.cxx
  src/RooExtendPdf.cxx
  src/RooExtendedBinding.cxx
  src/RooExtendedTerm.cxx
  src/RooFFTConvPdf.cxx
  src/RooFactoryWSTool.cxx
  src/RooFirstMoment.cxx
  src/RooFit/CodegenContext.cxx
  src/RooFit/EvalContext.cxx
  src/RooFit/Evaluator.cxx
  src/RooFitImplHelpers.cxx
  src/RooFitLegacy/RooCatTypeLegacy.cxx
  src/RooFitResult.cxx
  src/RooFoamGenerator.cxx
  src/RooFormulaUtils.cxx
  src/RooFormulaVar.cxx
  src/RooFracRemainder.cxx
  src/RooFunctor.cxx
  src/RooGenContext.cxx
  src/RooGenFitStudy.cxx
  src/RooGenProdProj.cxx
  src/RooGenericPdf.cxx
  src/RooGlobalFunc.cxx
  src/RooGrid.cxx
  src/RooHelpers.cxx
  src/RooHist.cxx
  src/RooHistError.cxx
  src/RooHistFunc.cxx
  src/RooHistPdf.cxx
  src/RooImproperIntegrator1D.cxx
  src/RooInvTransform.cxx
  src/RooLinTransBinning.cxx
  src/RooLinearCombination.cxx
  src/RooLinearVar.cxx
  src/RooLinkedList.cxx
  src/RooLinkedListElem.cxx
  src/RooMCIntegrator.cxx
  src/RooMCStudy.cxx
  src/RooMappedCategory.cxx
  src/RooMath.cxx
  src/RooMinimizer.cxx
  src/RooMinimizerFcn.cxx
  src/RooMoment.cxx
  src/RooMsgService.cxx
  src/RooMultiCategory.cxx
  src/RooMultiVarGaussian.cxx
  src/RooMultiPdf.cxx
  src/RooMultiReal.cxx
  src/RooNLLVarNew.cxx
  src/RooNameReg.cxx
  src/RooNormSetCache.cxx
  src/RooNormalizedPdf.cxx
  src/RooNumCdf.cxx
  src/RooNumConvPdf.cxx
  src/RooNumConvolution.cxx
  src/RooNumGenConfig.cxx
  src/RooNumGenFactory.cxx
  src/RooNumIntConfig.cxx
  src/RooNumIntFactory.cxx
  src/RooNumRunningInt.cxx
  src/RooObjCacheManager.cxx
  src/RooParamBinning.cxx
  src/RooPlot.cxx
  src/RooPlotable.cxx
  src/RooPolyFunc.cxx
  src/RooPolyVar.cxx
  src/RooPrintable.cxx
  src/RooProdGenContext.cxx
  src/RooProdPdf.cxx
  src/RooProduct.cxx
  src/RooProfileLL.cxx
  src/RooProjectedPdf.cxx
  src/RooPullVar.cxx
  src/RooQuasiRandomGenerator.cxx
  src/RooRandom.cxx
  src/RooRandomizeParamMCSModule.cxx
  src/RooRangeBinning.cxx
  src/RooRangeBoolean.cxx
  src/RooRatio.cxx
  src/RooRealBinding.cxx
  src/RooRealConstant.cxx
  src/RooRealIntegral.cxx
  src/RooRealSumFunc.cxx
  src/RooRealSumPdf.cxx
  src/RooRealVar.cxx
  src/RooRecursiveFraction.cxx
  src/RooResolutionModel.cxx
  src/RooRombergIntegrator.cxx
  src/RooSTLRefCountList.cxx
  src/RooSecondMoment.cxx
  src/RooSentinel.cxx
  src/RooSharedProperties.cxx
  src/RooSimGenContext.cxx
  src/RooSimSplitGenContext.cxx
  src/RooSimWSTool.cxx
  src/RooSimultaneous.cxx
  src/RooStreamParser.cxx
  src/RooStringVar.cxx
  src/RooStudyManager.cxx
  src/RooStudyPackage.cxx
  src/RooSuperCategory.cxx
  src/RooTFormulaEvaluator.cxx
  src/RooTObjWrap.cxx
  src/RooThresholdCategory.cxx
  src/RooTreeDataStore.cxx
  src/RooTruthModel.cxx
  src/RooUniformBinning.cxx
  src/RooUnitTest.cxx
  src/RooVectorDataStore.cxx
  src/RooWorkspace.cxx
  src/RooWrapperPdf.cxx
  src/TestStatistics/LikelihoodGradientWrapper.cxx
  src/TestStatistics/LikelihoodSerial.cxx
  src/TestStatistics/LikelihoodWrapper.cxx
  src/TestStatistics/RooAbsL.cxx
  src/TestStatistics/RooBinnedL.cxx
  src/TestStatistics/RooRealL.cxx
  src/TestStatistics/RooSubsidiaryL.cxx
  src/TestStatistics/RooSumL.cxx
  src/TestStatistics/RooUnbinnedL.cxx
  src/TestStatistics/buildLikelihood.cxx
  src/TestStatistics/SharedOffset.cxx
  ${LegacyEvalBackendSources}
  ${RooFitMPTestStatisticsSources}
PRIVATE
  FILE_SET private_header_files TYPE HEADERS BASE_DIRS src/ FILES
    src/RooConvIntegrandBinding.h
    src/RooFormulaUtils.h
    src/RooFormulaEvaluator.h
    src/RooTFormulaEvaluator.h
    src/RooFitLegacy/RooAbsCategoryLegacyIterator.h
    src/RooMCIntegrator.h
    src/RooFoamGenerator.h
    src/RooAbsMinimizerFcn.h
    src/RooSentinel.h
    src/RooAcceptReject.h
    src/TreeReadBuffer.h
    src/ConstraintHelpers.h
    src/RooRombergIntegrator.h
    src/RooBinIntegrator.h
    ${RooFitMPTestStatisticsHeaders}
    ${LegacyEvalBackendHeaders}
    src/RooFit/BatchModeDataHelpers.h
    src/RooMinimizerFcn.h
    src/RooAbsNumGenerator.h
    src/FitHelpers.h
    src/RooGenProdProj.h
    src/RooNumGenFactory.h
    src/RooGrid.h
    src/RooAddHelpers.h
    src/RooAdaptiveIntegratorND.h
    src/ValueChecking.h
    src/RooImproperIntegrator1D.h
  FILE_SET friend_header_files TYPE HEADERS BASE_DIRS res/ FILES
    res/RooFitImplHelpers.h
    res/RooUnitTest.h
PUBLIC
  FILE_SET HEADERS TYPE HEADERS BASE_DIRS inc/ FILES
    inc/Roo1DTable.h
    inc/RooAICRegistry.h
    inc/RooAbsAnaConvPdf.h
    inc/RooAbsArg.h
    inc/RooAbsBinning.h
    inc/RooAbsCache.h
    inc/RooAbsCacheElement.h
    inc/RooAbsCachedPdf.h
    inc/RooAbsCachedReal.h
    inc/RooAbsCategory.h
    inc/RooAbsCategoryLValue.h
    inc/RooAbsCollection.h
    inc/RooAbsData.h
    inc/RooAbsDataHelper.h
    inc/RooAbsDataStore.h
    inc/RooAbsFunc.h
    inc/RooAbsGenContext.h
    inc/RooAbsHiddenReal.h
    inc/RooAbsIntegrator.h
    inc/RooAbsLValue.h
    inc/RooAbsMCStudyModule.h
    inc/RooAbsMoment.h
    inc/RooAbsPdf.h
    inc/RooAbsProxy.h
    inc/RooAbsReal.h
    inc/RooAbsRealLValue.h
    inc/RooAbsSelfCachedPdf.h
    inc/RooAbsSelfCachedReal.h
    inc/RooAbsStudy.h
    inc/RooAddGenContext.h
    inc/RooAddModel.h
    inc/RooAddPdf.h
    inc/RooAddition.h
    inc/RooArgList.h
    inc/RooArgProxy.h
    inc/RooArgSet.h
    inc/RooBinSamplingPdf.h
    inc/RooBinWidthFunction.h
    inc/RooBinnedGenContext.h
    inc/RooBinning.h
    inc/RooBinningCategory.h
    inc/RooBrentRootFinder.h
    inc/RooCacheManager.h
    inc/RooCachedPdf.h
    inc/RooCachedReal.h
    inc/RooCategory.h
    inc/RooCategoryProxy.h
    inc/RooChangeTracker.h
    inc/RooClassFactory.h
    inc/RooCmdArg.h
    inc/RooCmdConfig.h
    inc/RooCollectionProxy.h
    inc/RooCompositeDataStore.h
    inc/RooConstVar.h
    inc/RooConstraintSum.h
    inc/RooConvCoefVar.h
    inc/RooConvGenContext.h
    inc/RooCurve.h
    inc/RooCustomizer.h
    inc/RooDLLSignificanceMCSModule.h
    inc/RooDataHist.h
    inc/RooDataHistSliceIter.h
    inc/RooDataProjBinding.h
    inc/RooDataSet.h
    inc/RooDerivative.h
    inc/RooDirItem.h
    inc/RooDouble.h
    inc/RooEffGenContext.h
    inc/RooEffProd.h
    inc/RooEfficiency.h
    inc/RooEllipse.h
    inc/RooErrorHandler.h
    inc/RooErrorVar.h
    inc/RooEvaluatorWrapper.h
    inc/RooExpensiveObjectCache.h
    inc/RooExtendPdf.h
    inc/RooExtendedBinding.h
    inc/RooExtendedTerm.h
    inc/RooFFTConvPdf.h
    inc/RooFactoryWSTool.h
    inc/RooFirstMoment.h
    inc/RooFit.h
    inc/RooFit/CodegenContext.h
    inc/RooFit/Config.h
    inc/RooFit/Detail/MathFuncs.h
    inc/RooFit/Detail/NormalizationHelpers.h
    inc/RooFit/Detail/RooNLLVarNew.h
    inc/RooFit/Detail/RooNormalizedPdf.h
    inc/RooFit/EvalContext.h
    inc/RooFit/Evaluator.h
    inc/RooFit/Floats.h
    inc/RooFit/ModelConfig.h
    inc/RooFit/TestStatistics/LikelihoodGradientWrapper.h
    inc/RooFit/TestStatistics/LikelihoodWrapper.h
    inc/RooFit/TestStatistics/RooAbsL.h
    inc/RooFit/TestStatistics/RooBinnedL.h
    inc/RooFit/TestStatistics/RooRealL.h
    inc/RooFit/TestStatistics/RooSubsidiaryL.h
    inc/RooFit/TestStatistics/RooSumL.h
    inc/RooFit/TestStatistics/RooUnbinnedL.h
    inc/RooFit/TestStatistics/SharedOffset.h
    inc/RooFit/TestStatistics/buildLikelihood.h
    inc/RooFit/UniqueId.h # Was not being included in ROOT_STL_PACKAGE call
    inc/RooFitLegacy/RooCatTypeLegacy.h
    inc/RooFitLegacy/RooCategorySharedProperties.h
    inc/RooFitLegacy/RooTreeData.h
    inc/RooFitResult.h
    inc/RooFormulaVar.h
    inc/RooFracRemainder.h
    inc/RooFunctor.h
    inc/RooGenContext.h
    inc/RooGenFitStudy.h
    inc/RooGenericPdf.h
    inc/RooGlobalFunc.h
    inc/RooHelpers.h
    inc/RooHist.h
    inc/RooHistError.h
    inc/RooHistFunc.h
    inc/RooHistPdf.h
    inc/RooInvTransform.h
    inc/RooLinTransBinning.h
    inc/RooLinearCombination.h
    inc/RooLinearVar.h
    inc/RooLinkedList.h
    inc/RooLinkedListElem.h
    inc/RooLinkedListIter.h
    inc/RooListProxy.h
    inc/RooMCStudy.h
    inc/RooMappedCategory.h
    inc/RooMath.h
    inc/RooMinimizer.h
    inc/RooMoment.h
    inc/RooMsgService.h
    inc/RooMultiCategory.h
    inc/RooMultiPdf.h
    inc/RooMultiReal.h
    inc/RooMultiVarGaussian.h
    inc/RooNameReg.h
    inc/RooNormSetCache.h
    inc/RooNumCdf.h
    inc/RooNumConvPdf.h
    inc/RooNumConvolution.h
    inc/RooNumGenConfig.h
    inc/RooNumIntConfig.h
    inc/RooNumIntFactory.h
    inc/RooNumRunningInt.h
    inc/RooNumber.h
    inc/RooObjCacheManager.h
    inc/RooParamBinning.h
    inc/RooPlot.h
    inc/RooPlotable.h
    inc/RooPolyFunc.h
    inc/RooPolyVar.h
    inc/RooPrintable.h
    inc/RooProdGenContext.h
    inc/RooProdPdf.h
    inc/RooProduct.h
    inc/RooProfileLL.h
    inc/RooProjectedPdf.h
    inc/RooPullVar.h
    inc/RooQuasiRandomGenerator.h
    inc/RooRandom.h
    inc/RooRandomizeParamMCSModule.h
    inc/RooRangeBinning.h
    inc/RooRangeBoolean.h
    inc/RooRatio.h
    inc/RooRealBinding.h
    inc/RooRealConstant.h
    inc/RooRealIntegral.h
    inc/RooRealProxy.h
    inc/RooRealSumFunc.h
    inc/RooRealSumPdf.h
    inc/RooRealVar.h
    inc/RooRealVarSharedProperties.h
    inc/RooRecursiveFraction.h
    inc/RooRefCountList.h
    inc/RooResolutionModel.h
    inc/RooSTLRefCountList.h
    inc/RooSecondMoment.h
    inc/RooSetProxy.h
    inc/RooSharedProperties.h
    inc/RooSimGenContext.h
    inc/RooSimSplitGenContext.h
    inc/RooSimWSTool.h
    inc/RooSimultaneous.h
    inc/RooStreamParser.h
    inc/RooStringVar.h
    inc/RooStringView.h
    inc/RooStudyManager.h
    inc/RooStudyPackage.h
    inc/RooSuperCategory.h
    inc/RooTObjWrap.h
    inc/RooTable.h
    inc/RooTemplateProxy.h
    inc/RooThresholdCategory.h
    inc/RooTreeDataStore.h
    inc/RooTruthModel.h
    inc/RooUniformBinning.h
    inc/RooVectorDataStore.h
    inc/RooWorkspace.h
    inc/RooWorkspaceHandle.h
    inc/RooWrapperPdf.h
  FILE_SET linkDef TYPE HEADERS BASE_DIRS inc/ FILES
    inc/LinkDef.h
)
target_link_libraries(RooFitCore
PRIVATE
  # Minuit2 is needed in the implementation of RooMinimizerFcn, to advertise
  # vanishing second derivatives to Minuit2Minimizer.
  Minuit2
  ${EXTRA_LIBRARIES}
PUBLIC
  Core
  Hist
  Graf
  Matrix
  Tree
  RIO
  MathCore
  Foam
  RooBatchCompute
  ${EXTRA_DEPENDENCIES}
)

# The following definitions are PUBLIC so they can also be used in ROOT-internal tests
if(roofit_legacy_eval_backend)
  target_compile_definitions(RooFitCore PUBLIC ROOFIT_LEGACY_EVAL_BACKEND)
endif()

if(NOT WIN32)
  target_compile_definitions(RooFitCore PUBLIC ROOFIT_MULTIPROCESS)
endif()

if(clad)
  target_compile_definitions(RooFitCore PUBLIC ROOFIT_CLAD)
endif()

if(cuda)
  target_compile_definitions(RooFitCore PUBLIC ROOFIT_CUDA)
endif()

if(fftw3)
  target_compile_definitions(RooFitCore PUBLIC ROOFIT_MATH_FFTW3)
  target_link_libraries(RooFitCore PRIVATE FFTW::Double)
endif()

# Used to silence deprecation warnings when testing deprecated features
target_compile_definitions(RooFitCore PUBLIC ROOFIT_BUILDS_ITSELF)

# The SMatrix package is header-only, but it's only exposed via the Smatrix
# target, which is a shared library that also includes the dictionaries. For
# RooFit, we are not interested in them, and want to avoid a dependency of the
# RooFitCore library on the Smatrix dictionary library. Therefore, we just add
# the include path.
target_include_directories(RooFitCore PRIVATE $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/math/smatrix/inc>)

# For recent clang, this can facilitate auto-vectorisation.
# In RooFit, the errno side effect is not needed, anyway:
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  target_compile_options(RooFitCore PUBLIC -fno-math-errno)
endif()

if (NOT WIN32)
  # workaround to include privately headers that are not public, from a "friend" target
  get_target_property(RooFitMultiProcess_Friend_Dirs RooFitMultiProcess HEADER_DIRS_friend_header_files)
  if(RooFitMultiProcess_Friend_Dirs)
    target_include_directories(RooFitCore PRIVATE ${RooFitMultiProcess_Friend_Dirs})
  endif()
endif()

ROOT_GENERATE_TARGET_DICTIONARY(RooFitCore "-writeEmptyRootPCM" ${EXTRA_DICT_OPTS})
ROOT_INSTALL_TARGET(RooFitCore)

ROOT_ADD_TEST_SUBDIRECTORY(test)
