# Module.mk for tmva module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODNAME      := tmva
MODDIR       := $(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TMVADIR      := $(MODDIR)
TMVADIRS     := $(TMVADIR)/src
TMVADIRI     := $(TMVADIR)/inc

##### libTMVA #####
TMVAL1       := $(MODDIRI)/LinkDef1.h
TMVAL2       := $(MODDIRI)/LinkDef2.h
TMVAL3       := $(MODDIRI)/LinkDef3.h
TMVAL4       := $(MODDIRI)/LinkDef4.h
TMVADS1      := $(MODDIRS)/G__TMVA1.cxx
TMVADS2      := $(MODDIRS)/G__TMVA2.cxx
TMVADS3      := $(MODDIRS)/G__TMVA3.cxx
TMVADS4      := $(MODDIRS)/G__TMVA4.cxx
TMVADO1      := $(TMVADS1:.cxx=.o)
TMVADO2      := $(TMVADS2:.cxx=.o)
TMVADO3      := $(TMVADS3:.cxx=.o)
TMVADO4      := $(TMVADS4:.cxx=.o)

TMVAL        := $(TMVAL1) $(TMVAL1) $(TMVAL3) $(TMVAL4)
TMVADS       := $(TMVAS1) $(TMVAS1) $(TMVAS3) $(TMVAS4)
TMVADO       := $(TMVAO1) $(TMVAO1) $(TMVAO3) $(TMVAO4)
TMVADH       := $(TMVADS:.cxx=.h)

TMVAH1       := Configurable.h Event.h Factory.h MethodBase.h MethodCompositeBase.h \
		MethodANNBase.h MethodTMlpANN.h MethodRuleFit.h MethodCuts.h MethodFisher.h \
		MethodKNN.h MethodCFMlpANN.h MethodCFMlpANN_Utils.h MethodLikelihood.h \
		MethodHMatrix.h MethodPDERS.h MethodBDT.h MethodDT.h MethodSVM.h MethodBayesClassifier.h \
		MethodFDA.h MethodMLP.h MethodCommittee.h MethodSeedDistance.h MethodBoost.h \
		MethodPDEFoam.h MethodLD.h MethodCategory.h
TMVAH2       := TSpline2.h TSpline1.h PDF.h BinaryTree.h BinarySearchTreeNode.h BinarySearchTree.h \
		Timer.h RootFinder.h CrossEntropy.h DecisionTree.h DecisionTreeNode.h MisClassificationError.h \
		Node.h SdivSqrtSplusB.h SeparationBase.h RegressionVariance.h Tools.h Reader.h \
		GeneticAlgorithm.h GeneticGenes.h GeneticPopulation.h GeneticRange.h GiniIndex.h \
		GiniIndexWithLaplace.h SimulatedAnnealing.h
TMVAH3       := Config.h KDEKernel.h Interval.h FitterBase.h MCFitter.h GeneticFitter.h SimulatedAnnealingFitter.h \
		MinuitFitter.h MinuitWrapper.h IFitterTarget.h IMetric.h MetricEuler.h MetricManhattan.h \
		SeedDistance.h PDEFoam.h PDEFoamDistr.h PDEFoamVect.h PDEFoamCell.h BDTEventWrapper.h CCTreeWrapper.h \
		CCPruner.h CostComplexityPruneTool.h SVEvent.h
TMVAH4       := TNeuron.h TSynapse.h TActivationChooser.h TActivation.h TActivationSigmoid.h TActivationIdentity.h \
		TActivationTanh.h TActivationRadial.h TNeuronInputChooser.h TNeuronInput.h TNeuronInputSum.h \
		TNeuronInputSqSum.h TNeuronInputAbs.h Types.h Ranking.h RuleFit.h RuleFitAPI.h IMethod.h MsgLogger.h \
		VariableTransformBase.h VariableIdentityTransform.h VariableDecorrTransform.h VariablePCATransform.h \
		VariableGaussTransform.h VariableNormalizeTransform.h
TMVAH1       := $(patsubst %,inc/%,$(TMVAH1))
TMVAH2       := $(patsubst %,inc/%,$(TMVAH2))
TMVAH3       := $(patsubst %,inc/%,$(TMVAH3))
TMVAH4       := $(patsubst %,inc/%,$(TMVAH4))
TMVAH        := $(TMVAH1) $(TMVAH2) $(TMVAH3) $(TMVAH4)
TMVAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TMVAO        := $(TMVAS:.cxx=.o)

TMVADEP      := $(TMVAO:.o=.d) $(TMVADO:.o=.d)

TMVALIB      := $(LPATH)/libTMVA.$(SOEXT)
TMVAMAP      := $(TMVALIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/TMVA/%.h,$(TMVAH))
ALLLIBS      += $(TMVALIB)
ALLMAPS      += $(TMVAMAP)

# include all dependency files
INCLUDEFILES += $(TMVADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/TMVA/%.h: $(TMVADIRI)/%.h
		@(if [ ! -d "include/TMVA" ]; then     \
		   mkdir -p include/TMVA;              \
		fi)
		cp $< $@

$(TMVALIB):     $(TMVAO) $(TMVADO) $(ORDER_) $(MAINLIBS) $(TMVALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTMVA.$(SOEXT) $@ "$(TMVAO) $(TMVADO)" \
		   "$(TMVALIBEXTRA)"

$(TMVADS1):     $(TMVAH1) $(TMVAL1) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TMVAH1) $(TMVAL1)
$(TMVADS2):     $(TMVAH2) $(TMVAL2) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TMVAH2) $(TMVAL2)
$(TMVADS3):     $(TMVAH3) $(TMVAL3) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TMVAH3) $(TMVAL3)
$(TMVADS4):     $(TMVAH4) $(TMVAL4) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TMVAH4) $(TMVAL4)

$(TMVAMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(TMVAL)
		$(RLIBMAP) -o $(TMVAMAP) -l $(TMVALIB) \
		   -d $(TMVALIBDEPM) -c $(TMVAL)

all-$(MODNAME): $(TMVALIB) $(TMVAMAP)

clean-$(MODNAME):
		@rm -f $(TMVADIRS)/*.o

clean::         clean-tmva

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TMVADEP) $(TMVADS) $(TMVADH) $(TMVALIB) $(TMVAMAP)
		@rm -rf include/TMVA

distclean::     distclean-$(MODNAME)
