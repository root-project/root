# Module.mk for tmva module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2009

MODNAME      := tmva
MODDIR       := $(ROOT_SRCDIR)/tmva/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TMVADIR      := $(MODDIR)
TMVADIRS     := $(TMVADIR)/src
TMVADIRI     := $(TMVADIR)/inc

##### libTMVA #####
TMVAL0       := $(MODDIRI)/LinkDef.h
TMVALS       := $(MODDIRI)/LinkDef1.h $(MODDIRI)/LinkDef2.h $(MODDIRI)/LinkDef3.h $(MODDIRI)/LinkDef4.h 
TMVADS       := $(call stripsrc,$(MODDIRS)/G__TMVA.cxx)
TMVADO       := $(TMVADS:.cxx=.o)
TMVADH       := $(TMVADS:.cxx=.h)

TMVAH1       := Configurable.h Event.h Factory.h MethodBase.h MethodCompositeBase.h \
		MethodANNBase.h MethodTMlpANN.h MethodRuleFit.h MethodCuts.h MethodFisher.h \
		MethodKNN.h MethodCFMlpANN.h MethodCFMlpANN_Utils.h MethodLikelihood.h \
		MethodHMatrix.h MethodPDERS.h MethodBDT.h MethodDT.h MethodSVM.h MethodBayesClassifier.h \
		MethodFDA.h MethodMLP.h MethodCommittee.h MethodBoost.h \
		MethodPDEFoam.h MethodLD.h MethodCategory.h MethodNN.h MethodDNN.h
TMVAH2       := TSpline2.h TSpline1.h PDF.h BinaryTree.h BinarySearchTreeNode.h BinarySearchTree.h \
		Timer.h RootFinder.h CrossEntropy.h DecisionTree.h DecisionTreeNode.h MisClassificationError.h \
		Node.h SdivSqrtSplusB.h SeparationBase.h RegressionVariance.h Tools.h Reader.h \
		GeneticAlgorithm.h GeneticGenes.h GeneticPopulation.h GeneticRange.h GiniIndex.h \
		GiniIndexWithLaplace.h SimulatedAnnealing.h 
TMVAH3       := Config.h KDEKernel.h Interval.h LogInterval.h FitterBase.h MCFitter.h GeneticFitter.h \
		SimulatedAnnealingFitter.h QuickMVAProbEstimator.h MinuitFitter.h MinuitWrapper.h IFitterTarget.h  \
		PDEFoam.h PDEFoamDecisionTree.h PDEFoamDensityBase.h PDEFoamDiscriminantDensity.h \
		PDEFoamEventDensity.h PDEFoamTargetDensity.h PDEFoamDecisionTreeDensity.h PDEFoamMultiTarget.h \
		PDEFoamVect.h PDEFoamCell.h PDEFoamDiscriminant.h PDEFoamEvent.h PDEFoamTarget.h \
		PDEFoamKernelBase.h PDEFoamKernelTrivial.h PDEFoamKernelLinN.h PDEFoamKernelGauss.h \
		BDTEventWrapper.h CCTreeWrapper.h \
		CCPruner.h CostComplexityPruneTool.h SVEvent.h OptimizeConfigParameters.h
TMVAH4       := TNeuron.h TSynapse.h TActivationChooser.h TActivation.h TActivationSigmoid.h TActivationIdentity.h \
		TActivationTanh.h TActivationReLU.h TActivationRadial.h TNeuronInputChooser.h TNeuronInput.h TNeuronInputSum.h \
		TNeuronInputSqSum.h TNeuronInputAbs.h Types.h Ranking.h RuleFit.h RuleFitAPI.h IMethod.h MsgLogger.h \
		VariableTransformBase.h VariableIdentityTransform.h VariableDecorrTransform.h VariablePCATransform.h \
		VariableGaussTransform.h VariableNormalizeTransform.h VariableRearrangeTransform.h ROCCalc.h
TMVADNN      :=  $(wildcard $(MODDIRI)/TMVA/DNN/*.h) $(wildcard $(MODDIRI)/TMVA/DNN/Architectures/*.h) \
		$(wildcard $(MODDIRI)/TMVA/DNN/Architectures/*/*.h)
ifneq ($(IMT),yes)
# FIXME: Add separate check if cuda was enabled.
TMVA_CUDAH    := $(MODDIRI)/TMVA/DNN/Architectures/Cuda.h $(wildcard $(MODDIRI)/TMVA/DNN/Architectures/Cuda/*)
TMVA_CPUH     := $(MODDIRI)/TMVA/DNN/Architectures/Cpu.h $(wildcard $(MODDIRI)/TMVA/DNN/Architectures/Cpu/*)
TMVADNN       := $(filter-out $(TMVA_CUDAH),$(TMVADNN))
TMVADNN       := $(filter-out $(TMVA_CPUH),$(TMVADNN))
endif

TMVAH1       := $(patsubst %,$(MODDIRI)/TMVA/%,$(TMVAH1))
TMVAH2       := $(patsubst %,$(MODDIRI)/TMVA/%,$(TMVAH2))
TMVAH3       := $(patsubst %,$(MODDIRI)/TMVA/%,$(TMVAH3))
TMVAH4       := $(patsubst %,$(MODDIRI)/TMVA/%,$(TMVAH4))

TMVAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/TMVA/*.*))
TMVAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TMVADNNS     := $(MODDIRS)/DNN/Architectures/Reference.cxx \
                $(MODDIRS)/DNN/Architectures/Reference/DataLoader.cxx
TMVAO        := $(call stripsrc,$(TMVAS:.cxx=.o))  $(call stripsrc,$(TMVADNNS:.cxx=.o))

TMVADEP      := $(TMVAO:.o=.d) $(TMVADO:.o=.d)

TMVALIB      := $(LPATH)/libTMVA.$(SOEXT)
TMVAMAP      := $(TMVALIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
TMVA_REL     := $(patsubst $(MODDIRI)/TMVA/%,include/TMVA/%,$(TMVAH) $(TMVADNN))
ALLHDRS      += $(TMVA_REL)
ALLLIBS      += $(TMVALIB)
ALLMAPS      += $(TMVAMAP)
TMVA_NOICC_REL := $(filter-out include/TMVA/NeuralNet.icc, $(TMVA_REL))
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(TMVA_NOICC_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Tmva_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(TMVALIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(TMVADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/TMVA/DNN/%.h: $(TMVADIRI)/TMVA/DNN/%.h
		@(if [ ! -d "include/TMVA/DNN/Architectures/Cpu" ]; then     \
		  mkdir -p include/TMVA/DNN/Architectures/Cpu;               \
		  mkdir -p include/TMVA/DNN/Architectures/Cuda;              \
		  mkdir -p include/TMVA/DNN/Architectures/Reference;         \
		fi)
		cp $< $@

include/TMVA/%.h: $(TMVADIRI)/TMVA/%.h
		@(if [ ! -d "include/TMVA" ]; then     \
		   mkdir -p include/TMVA;              \
		fi)
		cp $< $@

include/TMVA/%.icc: $(TMVADIRI)/TMVA/%.icc
		@(if [ ! -d "include/TMVA" ]; then     \
		   mkdir -p include/TMVA;              \
		fi)
		cp $< $@

$(TMVALIB):     $(TMVAO) $(TMVADO) $(ORDER_) $(MAINLIBS) $(TMVALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTMVA.$(SOEXT) $@ "$(TMVAO) $(TMVADO)" \
		   "$(OSTHREADLIBDIR) $(OSTHREADLIB) $(TMVALIBEXTRA)"

$(call pcmrule,TMVA)
	$(noop)

$(TMVADS):      $(TMVA_NOICC_REL) $(TMVAL0) $(TMVALS) $(ROOTCLINGEXE) $(call pcmdep,TMVA)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,TMVA) -c -writeEmptyRootPCM $(patsubst include/%,%,$(TMVA_NOICC_REL)) -I$(ROOT_SRCDIR) $(TMVAL0)

$(TMVAMAP):     $(TMVA_NOICC_REL) $(TMVAL0) $(TMVALS) $(ROOTCLINGEXE) $(call pcmdep,TMVA)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(TMVADS) $(call dictModule,TMVA) -c -I$(ROOT_SRCDIR) $(TMVA_NOICC_REL) $(TMVAL0)

all-$(MODNAME): $(TMVALIB)

clean-$(MODNAME):
		@rm -f $(TMVADIRS)/*.o

clean::         clean-tmva

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TMVADEP) $(TMVADS) $(TMVADH) $(TMVALIB) $(TMVAMAP)
		@rm -rf include/TMVA

distclean::     distclean-$(MODNAME)
