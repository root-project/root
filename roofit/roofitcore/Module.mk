# ROOT Module.mk for roofitcore module
# Copyright (c) 2005 Wouter Verkerke
#
# Author: Wouter Verkerke, 18/4/2005

MODNAME        := roofitcore
MODDIR         := $(ROOT_SRCDIR)/roofit/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

ROOFITCOREDIR  := $(MODDIR)
ROOFITCOREDIRS := $(ROOFITCOREDIR)/src
ROOFITCOREDIRI := $(ROOFITCOREDIR)/inc

##### libRooFitCore #####
ROOFITCOREL0   := $(MODDIRI)/LinkDef.h
ROOFITCORELS   := $(MODDIRI)/LinkDef1.h $(MODDIRI)/LinkDef2.h $(MODDIRI)/LinkDef3.h $(MODDIRI)/LinkDef4.h
ROOFITCOREDS   := $(call stripsrc,$(MODDIRS)/G__RooFitCore.cxx)
ROOFITCOREDO   := $(ROOFITCOREDS:.cxx=.o)
ROOFITCOREDH   := $(ROOFITCOREDS:.cxx=.h)

ROOFITCOREH1   := Roo1DTable.h RooAbsArg.h RooAbsBinning.h RooAbsCategory.h \
                  RooAbsCategoryLValue.h RooAbsCollection.h \
                  RooAbsData.h RooAbsFunc.h RooAbsGenContext.h \
                  RooAbsTestStatistic.h RooAbsHiddenReal.h RooAbsIntegrator.h \
                  RooAbsLValue.h RooAbsMCStudyModule.h RooAbsOptTestStatistic.h \
                  RooAbsPdf.h RooAbsProxy.h RooAbsReal.h \
                  RooAbsRealLValue.h RooAbsRootFinder.h RooAbsString.h \
                  RooAcceptReject.h RooAdaptiveGaussKronrodIntegrator1D.h \
                  RooAddGenContext.h RooAddition.h RooAddModel.h \
                  RooAICRegistry.h RooArgList.h RooArgProxy.h RooArgSet.h \
                  RooBanner.h RooBinning.h RooBinnedGenContext.h RooBrentRootFinder.h  RooCategory.h \
                  RooCategoryProxy.h RooCategorySharedProperties.h \
                  RooCatType.h RooChi2Var.h RooClassFactory.h RooCmdArg.h \
                  RooCmdConfig.h RooComplex.h RooConstVar.h RooConvCoefVar.h \
                  RooConvGenContext.h RooConvIntegrandBinding.h RooCurve.h \
                  RooCustomizer.h RooDataHist.h RooDataProjBinding.h RooDataSet.h \
                  RooDirItem.h RooDLLSignificanceMCSModule.h RooAbsAnaConvPdf.h \
                  RooAddPdf.h RooEfficiency.h RooEffProd.h RooExtendPdf.h

ROOFITCOREH2   := RooDouble.h RooEffGenContext.h RooEllipse.h RooErrorHandler.h \
                  RooErrorVar.h RooFit.h RooFitResult.h RooFormula.h \
                  RooFormulaVar.h RooGaussKronrodIntegrator1D.h \
                  RooGenContext.h RooGenericPdf.h RooGenProdProj.h RooGlobalFunc.h  \
                  RooGrid.h RooHashTable.h RooHistError.h \
                  RooHist.h RooImproperIntegrator1D.h \
                  RooBinIntegrator.h RooIntegrator1D.h RooIntegrator2D.h RooIntegratorBinding.h \
                  RooInt.h RooInvTransform.h RooLinearVar.h RooLinkedListElem.h \
                  RooLinkedList.h RooLinkedListIter.h RooLinTransBinning.h RooList.h \
                  RooListProxy.h RooMapCatEntry.h RooMappedCategory.h RooMath.h \
                  RooMCIntegrator.h RooMinuit.h RooMPSentinel.h \
                  RooMultiCategory.h RooMultiCatIter.h RooNameReg.h \
                  RooNameSet.h RooNLLVar.h RooNormSetCache.h RooNumber.h \
                  RooNumConvolution.h RooNumConvPdf.h RooNumIntConfig.h RooNumIntFactory.h \
                  RooPlotable.h RooPlot.h RooPolyVar.h RooPrintable.h \
                  RooProdGenContext.h RooProduct.h RooPullVar.h \
                  RooQuasiRandomGenerator.h RooRandom.h

ROOFITCOREH3   := RooRandomizeParamMCSModule.h RooRangeBinning.h RooRealAnalytic.h \
                  RooRealBinding.h RooRealConstant.h RooRealIntegral.h \
                  RooRealMPFE.h RooRealProxy.h RooRealVar.h \
                  RooRealVarSharedProperties.h RooRefCountList.h RooScaledFunc.h \
                  RooSegmentedIntegrator1D.h RooSegmentedIntegrator2D.h \
                  RooSetPair.h RooSetProxy.h RooSharedProperties.h \
                  RooSharedPropertiesList.h RooSimGenContext.h RooSimSplitGenContext.h \
                  RooStreamParser.h RooStringVar.h RooSuperCategory.h \
                  RooTable.h RooThreshEntry.h RooThresholdCategory.h \
                  RooTObjWrap.h RooTrace.h RooUniformBinning.h \
                  RooSimultaneous.h RooRealSumPdf.h RooResolutionModel.h \
                  RooProdPdf.h RooMCStudy.h RooSimPdfBuilder.h RooTruthModel.h RooMsgService.h \
                  RooProjectedPdf.h RooWorkspace.h RooProfileLL.h RooAbsCachedPdf.h RooAbsSelfCachedPdf.h \
                  RooHistPdf.h RooCachedPdf.h RooFFTConvPdf.h RooDataHistSliceIter.h RooCacheManager.h \
                  RooAbsCache.h RooAbsCacheElement.h RooObjCacheManager.h RooExtendedTerm.h RooSentinel.h \
                  RooParamBinning.h 

ROOFITCOREH4   := RooConstraintSum.h RooRecursiveFraction.h RooDataWeightedAverage.h \
                  RooSimWSTool.h RooFracRemainder.h RooAbsCachedReal.h \
                  RooAbsSelfCachedReal.h RooCachedReal.h RooNumCdf.h RooChangeTracker.h \
                  RooNumRunningInt.h RooHistFunc.h RooExpensiveObjectCache.h \
                  RooBinningCategory.h RooCintUtils.h RooFactoryWSTool.h RooTFoamBinding.h RooFunctor.h	\
                  RooDerivative.h RooGenFunction.h RooMultiGenFunction.h RooAdaptiveIntegratorND.h \
                  RooAbsNumGenerator.h RooFoamGenerator.h RooNumGenConfig.h RooNumGenFactory.h \
                  RooMultiVarGaussian.h RooXYChi2Var.h RooAbsDataStore.h RooTreeDataStore.h RooTreeData.h \
                  RooMinimizer.h RooMinimizerFcn.h RooMoment.h RooStudyManager.h RooAbsStudy.h \
                  RooGenFitStudy.h RooProofDriverSelector.h RooStudyPackage.h RooCompositeDataStore.h \
		  RooRangeBoolean.h RooVectorDataStore.h RooUnitTest.h RooExtendedBinding.h \
                  RooAbsMoment.h RooFirstMoment.h RooSecondMoment.h

ROOFITCOREH1   := $(patsubst %,$(MODDIRI)/%,$(ROOFITCOREH1))
ROOFITCOREH2   := $(patsubst %,$(MODDIRI)/%,$(ROOFITCOREH2))
ROOFITCOREH3   := $(patsubst %,$(MODDIRI)/%,$(ROOFITCOREH3))
ROOFITCOREH4   := $(patsubst %,$(MODDIRI)/%,$(ROOFITCOREH4))
ROOFITCOREH    := $(ROOFITCOREH1) $(ROOFITCOREH2) $(ROOFITCOREH3) $(ROOFITCOREH4)
ROOFITCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ROOFITCOREO    := $(call stripsrc,$(ROOFITCORES:.cxx=.o))

ROOFITCOREDEP  := $(ROOFITCOREO:.o=.d) $(ROOFITCOREDO:.o=.d)

ROOFITCORELIB  := $(LPATH)/libRooFitCore.$(SOEXT)
ROOFITCOREMAP  := $(ROOFITCORELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS        += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOFITCOREH))
ALLLIBS        += $(ROOFITCORELIB)
ALLMAPS        += $(ROOFITCOREMAP)

# include all dependency files
INCLUDEFILES   += $(ROOFITCOREDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h: $(ROOFITCOREDIRI)/%.h
		cp $< $@

$(ROOFITCORELIB): $(ROOFITCOREO) $(ROOFITCOREDO) $(ORDER_) $(MAINLIBS) \
                  $(ROOFITCORELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRooFitCore.$(SOEXT) $@ \
		   "$(ROOFITCOREO) $(ROOFITCOREDO)" \
		   "$(ROOFITCORELIBEXTRA) $(OSTHREADLIBDIR) $(OSTHREADLIB)"

$(call pcmrule,ROOFITCORE)
	$(noop)

$(ROOFITCOREDS): $(ROOFITCOREH) $(ROOFITCOREL0) $(ROOFITCORELS) $(ROOTCLINGEXE) $(call pcmdep,ROOFITCORE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,ROOFITCORE) -c -writeEmptyRootPCM -I$(ROOT_SRCDIR) $(ROOFITCOREH) $(ROOFITCOREL0)

$(ROOFITCOREMAP): $(ROOFITCOREH) $(ROOFITCOREL0) $(ROOFITCORELS) $(ROOTCLINGEXE) $(call pcmdep,ROOFITCORE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(ROOFITCOREDS) $(call dictModule,ROOFITCORE) -c -I$(ROOT_SRCDIR) $(ROOFITCOREH) $(ROOFITCOREL0)

all-$(MODNAME): $(ROOFITCORELIB)

clean-$(MODNAME):
		@rm -f $(ROOFITCOREO) $(ROOFITCOREDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -rf $(ROOFITCOREDEP) $(ROOFITCORELIB) $(ROOFITCOREMAP) \
		   $(ROOFITCOREDS) $(ROOFITCOREDH)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(ROOFITCOREDO): NOOPT = $(OPT)
# FIXME: Temporarily until we understand where the errors come from.
$(ROOFITCOREDO): CXXFLAGS := $(filter-out -Xclang -fmodules -Xclang -fmodules-cache-path=$(ROOTSYS)/pcm/, $(CXXFLAGS))
