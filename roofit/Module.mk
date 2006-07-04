# ROOT Module.mk for roofit module
# Copyright (c) 2005 Wouter Verkerke
#
# Author: Wouter Verkerke, 18/4/2005

MODDIR       := roofit
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ROOFITDIR    := $(MODDIR)
ROOFITDIRS   := $(ROOFITDIR)/src
ROOFITDIRI   := $(ROOFITDIR)/inc

ROOFITVERS   := roofit_2.09
ROOFITSRCS   := $(MODDIR)/$(ROOFITVERS).src.tgz
ROOFITETAG   := $(MODDIR)/headers.d

##### libRooFit #####
ROOFITL1     := $(MODDIRI)/LinkDef1.h
ROOFITL2     := $(MODDIRI)/LinkDef2.h
ROOFITL3     := $(MODDIRI)/LinkDef3.h
ROOFITDS1    := $(MODDIRS)/G__Roofit1.cxx
ROOFITDS2    := $(MODDIRS)/G__Roofit2.cxx
ROOFITDS3    := $(MODDIRS)/G__Roofit3.cxx
ROOFITDO1    := $(ROOFITDS1:.cxx=.o)
ROOFITDO2    := $(ROOFITDS2:.cxx=.o)
ROOFITDO3    := $(ROOFITDS3:.cxx=.o)
ROOFITDS     := $(ROOFITDS1) $(ROOFITDS2) $(ROOFITDS3)
ROOFITDO     := $(ROOFITDO1) $(ROOFITDO2) $(ROOFITDO3)
ROOFITDH     := $(ROOFITDS:.cxx=.h)

ROOFITH1     := Roo1DTable.h Roo2DKeysPdf.h RooAbsAnaConvPdf.h RooAbsArg.h RooAbsBinning.h RooAbsCategory.h RooAbsCategoryLValue.h \
                RooAbsCollection.h RooAbsData.h RooAbsFunc.h RooAbsGenContext.h RooAbsGoodnessOfFit.h RooAbsHiddenReal.h \
                RooAbsIntegrator.h RooAbsLValue.h RooAbsOptGoodnessOfFit.h RooAbsPdf.h RooAbsProxy.h RooAbsReal.h RooAbsRealLValue.h \
                RooAbsRootFinder.h RooAbsString.h RooAcceptReject.h RooAdaptiveGaussKronrodIntegrator1D.h RooAddGenContext.h \
                RooAddition.h RooAddModel.h RooAddPdf.h RooAICRegistry.h RooArgList.h RooArgProxy.h RooArgSet.h RooArgusBG.h \
                RooBCPEffDecay.h RooBCPGenDecay.h RooBDecay.h RooBifurGauss.h RooBinning.h RooBlindTools.h RooBMixDecay.h \
                RooBreitWigner.h RooBrentRootFinder.h RooBukinPdf.h RooCategory.h RooCategoryProxy.h RooCategorySharedProperties.h \
                RooCatType.h RooCBShape.h RooChebychev.h RooChi2Var.h RooClassFactory.h RooCmdArg.h RooCmdConfig.h RooComplex.h \
                RooConstVar.h RooConvCoefVar.h RooConvGenContext.h RooConvIntegrandBinding.h RooCurve.h RooCustomizer.h \
                 RooGlobalFunc.h
ROOFITH2     := RooDataHist.h RooDataProjBinding.h RooDataSet.h RooDecay.h RooDirItem.h RooDouble.h RooDstD0BG.h RooEffGenContext.h \
                RooEfficiency.h RooEffProd.h RooEllipse.h RooErrorHandler.h RooErrorVar.h RooExponential.h RooExtendPdf.h \
                RooFitResult.h RooFormula.h RooFormulaVar.h RooGaussian.h RooGaussKronrodIntegrator1D.h RooGaussModel.h \
                RooGenCategory.h RooGenContext.h RooGenericPdf.h RooGenProdProj.h RooGExpModel.h RooGraphEdge.h RooGraphNode.h \
                RooGraphSpring.h RooGrid.h RooHashTable.h RooHistError.h RooHist.h RooHistPdf.h RooHtml.h RooImproperIntegrator1D.h \
                RooIntegrator1D.h RooIntegrator2D.h RooIntegratorBinding.h RooInt.h RooInvTransform.h RooKeysPdf.h \
                RooLandau.h RooLinearVar.h RooLinkedListElem.h RooLinkedList.h RooLinkedListIter.h RooLinTransBinning.h \
                RooList.h RooListProxy.h RooMapCatEntry.h RooMappedCategory.h RooMath.h RooMCIntegrator.h RooMCStudy.h \
                RooMinuit.h RooMPSentinel.h RooMultiCategory.h RooMultiCatIter.h RooNameReg.h  RooFit.h
ROOFITH3     := RooNameSet.h RooNLLVar.h RooNonCPEigenDecay.h RooNormListManager.h RooNormManager.h RooNormSetCache.h \
                RooNovosibirsk.h RooNumber.h RooNumConvolution.h RooNumConvPdf.h RooNumIntConfig.h RooNumIntFactory.h \
                RooParametricStepFunction.h RooPlotable.h RooPlot.h RooPolynomial.h RooPolyVar.h RooPrintable.h RooProdGenContext.h \
                RooProdPdf.h RooProduct.h RooPullVar.h RooQuasiRandomGenerator.h RooRandom.h RooRangeBinning.h RooRealAnalytic.h \
                RooRealBinding.h RooRealConstant.h RooRealIntegral.h RooRealMPFE.h RooRealProxy.h RooRealSumPdf.h RooRealVar.h \
                RooRealVarSharedProperties.h RooRefCountList.h RooResolutionModel.h RooScaledFunc.h RooSegmentedIntegrator1D.h \
                RooSegmentedIntegrator2D.h RooSetPair.h RooSetProxy.h RooSharedProperties.h RooSharedPropertiesList.h \
                RooSimGenContext.h RooSimPdfBuilder.h RooSimultaneous.h RooStreamParser.h RooStringVar.h RooSuperCategory.h \
                RooTable.h RooThreshEntry.h RooThresholdCategory.h RooTObjWrap.h RooTrace.h RooTreeData.h RooTruthModel.h \
                RooUnblindCPAsymVar.h RooUnblindOffset.h RooUnblindPrecision.h RooUnblindUniform.h RooUniformBinning.h \
                RooVoigtian.h 

ROOFITH1     := $(patsubst %,$(MODDIRI)/%,$(ROOFITH1))
ROOFITH2     := $(patsubst %,$(MODDIRI)/%,$(ROOFITH2))
ROOFITH3     := $(patsubst %,$(MODDIRI)/%,$(ROOFITH3))
ROOFITH      := $(ROOFITH1) $(ROOFITH2) $(ROOFITH3)
ROOFITS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ROOFITO      := $(ROOFITS:.cxx=.o)
ROOFITO      := $(ROOFITS:.cxx=.o)

ROOFITDEP    := $(ROOFITO:.o=.d) $(ROOFITDO:.o=.d)
ROOFITLIB    := $(LPATH)/libRooFit.$(SOEXT)

# used in the main Makefile
ROOFITINCH   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(ROOFITH))
ALLHDRS      += $(ROOFITINCH)
ALLLIBS      += $(ROOFITLIB)

# include all dependency files
ifeq ($(findstring $(MAKECMDGOALS),clean-roofit distclean-roofit),)
INCLUDEFILES += $(ROOFITDEP)
endif

##### local rules #####
$(ROOFITETAG): $(ROOFITSRCS)
		@(if [ ! -d src ]; then \
		   echo "*** Extracting roofit source ..."; \
		   cd roofit ; \
		   if [ "x`which gtar 2>/dev/null | awk '{if ($$1~/gtar/) print $$1;}'`" != "x" ]; then \
		      gtar xzf $(ROOFITVERS).src.tgz; \
		   else \
		      gunzip -c $(ROOFITVERS).src.tgz | tar xf -; \
		   fi; \
		   etag=`basename $(ROOFITETAG)` ; \
		   touch $$etag ; \
		fi)

$(ROOFITH):     $(ROOFITETAG)

# Use static rule here instead of implicit rule
$(ROOFITINCH):  include/%.h: $(ROOFITDIRI)/%.h
		cp $< $@

$(ROOFITLIB):   $(ROOFITO) $(ROOFITDO) $(MAINLIBS) $(ROOFITLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRooFit.$(SOEXT) $@ "$(ROOFITO) $(ROOFITDO)" \
		   "$(ROOFITLIBEXTRA)"

$(ROOFITDS1):   $(ROOFITETAG) $(ROOFITH1) $(ROOFITL1) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOFITH1) $(ROOFITL1)

$(ROOFITDS2):   $(ROOFITETAG) $(ROOFITH2) $(ROOFITL2) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOFITH2) $(ROOFITL2)

$(ROOFITDS3):   $(ROOFITETAG) $(ROOFITH3) $(ROOFITL3) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(ROOFITH3) $(ROOFITL3)

$(ROOFITDO1):   $(ROOFITDS1)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

$(ROOFITDO2):   $(ROOFITDS2)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

$(ROOFITDO3):   $(ROOFITDS3)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-roofit:     $(ROOFITLIB)

map-roofit:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(ROOFITLIB) \
		   -d $(ROOFITLIBDEP) -c $(ROOFITL1) $(ROOFITL2) $(ROOFITL3)

map::           map-roofit

clean-roofit:
		@rm -f $(ROOFITO) $(ROOFITDO)

clean::         clean-roofit

distclean-roofit: clean-roofit
		@rm -rf $(ROOFITETAG) $(ROOFITDEP) $(ROOFITLIB) \
		   $(ROOFITDIRS) $(ROOFITDIRI)

distclean::     distclean-roofit
