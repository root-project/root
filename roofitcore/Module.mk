# ROOT Module.mk for roofitcore module
# Copyright (c) 2005 Wouter Verkerke
#
# Author: Wouter Verkerke, 18/4/2005

MODDIR         := roofitcore
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

ROOFITCOREDIR  := $(MODDIR)
ROOFITCOREDIRS := $(ROOFITCOREDIR)/src
ROOFITCOREDIRI := $(ROOFITCOREDIR)/inc

##### libRooFitCore #####
ROOFITCOREL1   := $(MODDIRI)/LinkDef1.h
ROOFITCOREL2   := $(MODDIRI)/LinkDef2.h
ROOFITCOREL3   := $(MODDIRI)/LinkDef3.h
ROOFITCOREDS1  := $(MODDIRS)/G__RooFitCore1.cxx
ROOFITCOREDS2  := $(MODDIRS)/G__RooFitCore2.cxx
ROOFITCOREDS3  := $(MODDIRS)/G__RooFitCore3.cxx
ROOFITCOREDO1  := $(ROOFITCOREDS1:.cxx=.o)
ROOFITCOREDO2  := $(ROOFITCOREDS2:.cxx=.o)
ROOFITCOREDO3  := $(ROOFITCOREDS3:.cxx=.o)
ROOFITCOREL    := $(ROOFITCOREL1) $(ROOFITCOREL2) $(ROOFITCOREL3)
ROOFITCOREDS   := $(ROOFITCOREDS1) $(ROOFITCOREDS2) $(ROOFITCOREDS3)
ROOFITCOREDO   := $(ROOFITCOREDO1) $(ROOFITCOREDO2) $(ROOFITCOREDO3)
ROOFITCOREDH   := $(ROOFITCOREDS:.cxx=.h)

ROOFITCOREH1   := Roo1DTable.h RooAbsArg.h RooAbsBinning.h RooAbsCategory.h \
                  RooAbsCategoryLValue.h RooAbsCollection.h \
                  RooAbsData.h RooAbsFunc.h RooAbsGenContext.h \
                  RooAbsGoodnessOfFit.h RooAbsHiddenReal.h RooAbsIntegrator.h \
                  RooAbsLValue.h RooAbsMCStudyModule.h RooAbsOptGoodnessOfFit.h \
                  RooAbsPdf.h RooAbsProxy.h RooAbsReal.h \
                  RooAbsRealLValue.h RooAbsRootFinder.h RooAbsString.h \
                  RooAcceptReject.h RooAdaptiveGaussKronrodIntegrator1D.h \
                  RooAddGenContext.h RooAddition.h RooAddModel.h \
                  RooAICRegistry.h RooArgList.h RooArgProxy.h RooArgSet.h \
                  RooBanner.h RooBinning.h RooBrentRootFinder.h  RooCategory.h \
                  RooCategoryProxy.h RooCategorySharedProperties.h \
                  RooCatType.h RooChi2Var.h RooClassFactory.h RooCmdArg.h \
                  RooCmdConfig.h RooComplex.h RooConstVar.h RooConvCoefVar.h \
                  RooConvGenContext.h RooConvIntegrandBinding.h RooCurve.h \
                  RooCustomizer.h RooDataHist.h RooDataProjBinding.h RooDataSet.h \
                  RooDirItem.h RooDLLSignificanceMCSModule.h RooAbsAnaConvPdf.h \
                  RooAddPdf.h RooEfficiency.h RooEffProd.h RooExtendPdf.h

ROOFITCOREH2   := RooDouble.h RooEffGenContext.h RooEllipse.h RooErrorHandler.h \
                  RooErrorVar.h RooFit.h RooFitResult.h RooFormula.h \
                  RooFormulaVar.h RooGaussKronrodIntegrator1D.h RooGenCategory.h \
                  RooGenContext.h RooGenericPdf.h RooGenProdProj.h \
                  RooGlobalFunc.h RooGraphEdge.h RooGraphNode.h RooGraphSpring.h \
                  RooGrid.h RooHashTable.h RooHistError.h \
                  RooHist.h RooHtml.h RooImproperIntegrator1D.h \
                  RooIntegrator1D.h RooIntegrator2D.h RooIntegratorBinding.h \
                  RooInt.h RooInvTransform.h RooLinearVar.h RooLinkedListElem.h \
                  RooLinkedList.h RooLinkedListIter.h RooLinTransBinning.h RooList.h \
                  RooListProxy.h RooMapCatEntry.h RooMappedCategory.h RooMath.h \
                  RooMCIntegrator.h RooMinuit.h RooMPSentinel.h \
                  RooMultiCategory.h RooMultiCatIter.h RooNameReg.h \
                  RooNameSet.h RooNLLVar.h RooNormListManager.h \
                  RooNormManager.h RooNormSetCache.h RooNumber.h \
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
                  RooSharedPropertiesList.h RooSimGenContext.h \
                  RooStreamParser.h RooStringVar.h RooSuperCategory.h \
                  RooTable.h RooThreshEntry.h RooThresholdCategory.h \
                  RooTObjWrap.h RooTrace.h RooTreeData.h RooUniformBinning.h \
                  RooSimultaneous.h RooRealSumPdf.h RooResolutionModel.h \
                  RooProdPdf.h RooMCStudy.h RooSimPdfBuilder.h RooTruthModel.h

ROOFITCOREH1   := $(patsubst %,$(MODDIRI)/%,$(ROOFITCOREH1))
ROOFITCOREH2   := $(patsubst %,$(MODDIRI)/%,$(ROOFITCOREH2))
ROOFITCOREH3   := $(patsubst %,$(MODDIRI)/%,$(ROOFITCOREH3))
ROOFITCOREH    := $(ROOFITCOREH1) $(ROOFITCOREH2) $(ROOFITCOREH3)
ROOFITCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ROOFITCOREO    := $(ROOFITCORES:.cxx=.o)

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
include/%.h: $(ROOFITCOREDIRI)/%.h
		cp $< $@

$(ROOFITCORELIB): $(ROOFITCOREO) $(ROOFITCOREDO) $(ORDER_) $(MAINLIBS) $(ROOFITCORELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRooFitCore.$(SOEXT) $@ "$(ROOFITCOREO) $(ROOFITCOREDO)" \
		   "$(ROOFITCORELIBEXTRA)"

$(ROOFITCOREDS1): $(ROOFITCOREH1) $(ROOFITCOREL1) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOFITCOREH1) $(ROOFITCOREL1)

$(ROOFITCOREDS2): $(ROOFITCOREH2) $(ROOFITCOREL2) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOFITCOREH2) $(ROOFITCOREL2)

$(ROOFITCOREDS3): $(ROOFITCOREH3) $(ROOFITCOREL3) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOFITCOREH3) $(ROOFITCOREL3)

$(ROOFITCOREMAP): $(RLIBMAP) $(MAKEFILEDEP) $(ROOFITCOREL)
		$(RLIBMAP) -o $(ROOFITCOREMAP) -l $(ROOFITCORELIB) \
		   -d $(ROOFITCORELIBDEPM) -c $(ROOFITCOREL)

all-roofitcore: $(ROOFITCORELIB) $(ROOFITCOREMAP)

clean-roofitcore:
		@rm -f $(ROOFITCOREO) $(ROOFITCOREDO)

clean::         clean-roofitcore

distclean-roofitcore: clean-roofitcore
		@rm -rf $(ROOFITCOREDEP) $(ROOFITCORELIB) $(ROOFITCOREMAP) \
		   $(ROOFITCOREDS) $(ROOFITCOREDH)

distclean::     distclean-roofitcore
