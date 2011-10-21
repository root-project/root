############################################################################
# CMakeLists.txt file for building ROOT roofitcore package
# @author Pere Mato, CERN
############################################################################



set(headers1 Roo1DTable.h RooAbsArg.h RooAbsBinning.h RooAbsCategory.h RooAbsCategoryLValue.h RooAbsCollection.h
             RooAbsData.h RooAbsFunc.h RooAbsGenContext.h RooAbsTestStatistic.h RooAbsHiddenReal.h RooAbsIntegrator.h 
             RooAbsLValue.h RooAbsMCStudyModule.h RooAbsOptTestStatistic.h RooAbsPdf.h RooAbsProxy.h RooAbsReal.h
             RooAbsRealLValue.h RooAbsRootFinder.h RooAbsString.h RooAcceptReject.h RooAdaptiveGaussKronrodIntegrator1D.h
             RooAddGenContext.h RooAddition.h RooAddModel.h RooAICRegistry.h RooArgList.h RooArgProxy.h RooArgSet.h 
             RooBanner.h RooBinning.h RooBrentRootFinder.h  RooCategory.h RooCategoryProxy.h RooCategorySharedProperties.h
             RooCatType.h RooChi2Var.h RooClassFactory.h RooCmdArg.h RooCmdConfig.h RooComplex.h RooConstVar.h RooConvCoefVar.h
             RooConvGenContext.h RooConvIntegrandBinding.h RooCurve.h RooCustomizer.h RooDataHist.h RooDataProjBinding.h RooDataSet.h
             RooDirItem.h RooDLLSignificanceMCSModule.h RooAbsAnaConvPdf.h RooAddPdf.h RooEfficiency.h RooEffProd.h RooExtendPdf.h )
set(headers2 RooDouble.h RooEffGenContext.h RooEllipse.h RooErrorHandler.h RooErrorVar.h RooFit.h RooFitResult.h RooFormula.h
             RooFormulaVar.h RooGaussKronrodIntegrator1D.h RooGenCategory.h RooGenContext.h RooGenericPdf.h RooGenProdProj.h RooGlobalFunc.h
             RooGrid.h RooHashTable.h RooHistError.h RooHist.h RooImproperIntegrator1D.h
             RooIntegrator1D.h RooIntegrator2D.h RooIntegratorBinding.h RooInt.h RooInvTransform.h RooLinearVar.h RooLinkedListElem.h
             RooLinkedList.h RooLinkedListIter.h RooLinTransBinning.h RooList.h RooListProxy.h RooMapCatEntry.h RooMappedCategory.h RooMath.h
             RooMCIntegrator.h RooMinuit.h RooMPSentinel.h RooMultiCategory.h RooMultiCatIter.h RooNameReg.h
             RooNameSet.h RooNLLVar.h RooNormSetCache.h RooNumber.h RooNumConvolution.h RooNumConvPdf.h RooNumIntConfig.h RooNumIntFactory.h
             RooPlotable.h RooPlot.h RooPolyVar.h RooPrintable.h RooProdGenContext.h RooProduct.h RooPullVar.h
             RooQuasiRandomGenerator.h RooRandom.h)
set(headers3 RooRandomizeParamMCSModule.h RooRangeBinning.h RooRealAnalytic.h RooRealBinding.h RooRealConstant.h RooRealIntegral.h 
             RooRealMPFE.h RooRealProxy.h RooRealVar.h RooRealVarSharedProperties.h RooRefCountList.h RooScaledFunc.h
             RooSegmentedIntegrator1D.h RooSegmentedIntegrator2D.h RooSetPair.h RooSetProxy.h RooSharedProperties.h
             RooSharedPropertiesList.h RooSimGenContext.h RooStreamParser.h RooStringVar.h RooSuperCategory.h
             RooTable.h RooThreshEntry.h RooThresholdCategory.h RooTObjWrap.h RooTrace.h RooUniformBinning.h
             RooSimultaneous.h RooRealSumPdf.h RooResolutionModel.h RooProdPdf.h RooMCStudy.h RooSimPdfBuilder.h RooTruthModel.h RooMsgService.h
             RooProjectedPdf.h RooWorkspace.h RooProfileLL.h RooAbsCachedPdf.h RooAbsSelfCachedPdf.h
             RooHistPdf.h RooCachedPdf.h RooFFTConvPdf.h RooDataHistSliceIter.h RooCacheManager.h
             RooAbsCache.h RooAbsCacheElement.h RooObjCacheManager.h RooExtendedTerm.h RooSentinel.h
             RooParamBinning.h RooConstraintSum.h RooRecursiveFraction.h RooDataWeightedAverage.h
             RooSimWSTool.h RooFracRemainder.h RooAbsCachedReal.h RooAbsSelfCachedReal.h RooCachedReal.h RooNumCdf.h RooChangeTracker.h
             RooNumRunningInt.h RooHistFunc.h RooExpensiveObjectCache.h
             RooBinningCategory.h RooCintUtils.h RooFactoryWSTool.h RooTFoamBinding.h RooFunctor.h
             RooDerivative.h RooGenFunction.h RooMultiGenFunction.h RooAdaptiveIntegratorND.h
             RooAbsNumGenerator.h RooFoamGenerator.h RooNumGenConfig.h RooNumGenFactory.h 
             RooMultiVarGaussian.h RooXYChi2Var.h RooAbsDataStore.h RooTreeDataStore.h RooTreeData.h
             RooMinimizer.h RooMinimizerFcn.h RooMoment.h RooStudyManager.h RooAbsStudy.h
             RooGenFitStudy.h RooProofDriverSelector.h RooStudyPackage.h RooCompositeDataStore.h RooRangeBoolean.h RooVectorDataStore.h)

ROOT_GENERATE_DICTIONARY(G__RooFitCore1 ${headers1} LINKDEF LinkDef1.h)
ROOT_GENERATE_DICTIONARY(G__RooFitCore2 ${headers2} LINKDEF LinkDef2.h)
ROOT_GENERATE_DICTIONARY(G__RooFitCore3 ${headers3} LINKDEF LinkDef3.h)

ROOT_GENERATE_ROOTMAP(RooFitCore LINKDEF LinkDef1.h LinkDef2.h LinkDef3.h
                                 DEPENDENCIES Hist Graf Matrix Tree Minuit RIO MathCore Foam )
ROOT_LINKER_LIBRARY(RooFitCore *.cxx G__RooFitCore1.cxx G__RooFitCore2.cxx G__RooFitCore3.cxx CMAKENOEXPORT  LIBRARIES Core Cint 
                    DEPENDENCIES Hist Graf Matrix Tree Minuit RIO MathCore Foam)
ROOT_INSTALL_HEADERS()

