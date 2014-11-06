# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODNAME      := mathcore
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATHCOREDIR  := $(MODDIR)
MATHCOREDIRS := $(MATHCOREDIR)/src
MATHCOREDIRI := $(MATHCOREDIR)/inc
MATHCOREDIRT := $(call stripsrc,$(MATHCOREDIR)/test)

##### libMathCore #####
MATHCOREL0   := $(MODDIRI)/LinkDef.h
MATHCORELS   := $(MODDIRI)/LinkDef1.h $(MODDIRI)/LinkDef2.h $(MODDIRI)/LinkDef_Func.h $(MODDIRI)/LinkDef3.h
MATHCOREDS   := $(call stripsrc,$(MODDIRS)/G__MathCore.cxx)
MATHCOREDO   := $(MATHCOREDS:.cxx=.o)
MATHCOREDH   := $(MATHCOREDS:.cxx=.h)

MATHCOREDH1  := $(MODDIRI)/TComplex.h \
                $(MODDIRI)/TMath.h 
MATHCOREDH2  := $(MODDIRI)/TRandom.h \
                $(MODDIRI)/TRandom1.h \
                $(MODDIRI)/TRandom2.h \
                $(MODDIRI)/TRandom3.h \
                $(MODDIRI)/TStatistic.h \
                $(MODDIRI)/TKDTree.h \
                $(MODDIRI)/TKDTreeBinning.h \
                $(MODDIRI)/Math/KDTree.h \
                $(MODDIRI)/Math/TDataPoint.h \
                $(MODDIRI)/Math/TDataPointN.h \
                $(MODDIRI)/Math/IParamFunction.h \
                $(MODDIRI)/Math/IFunction.h \
                $(MODDIRI)/Math/ParamFunctor.h \
                $(MODDIRI)/Math/Functor.h \
                $(MODDIRI)/Math/Minimizer.h \
                $(MODDIRI)/Math/MinimizerOptions.h \
                $(MODDIRI)/Math/MinimTransformFunction.h \
                $(MODDIRI)/Math/MinimTransformVariable.h \
                $(MODDIRI)/Math/BasicMinimizer.h \
                $(MODDIRI)/Math/IntegratorOptions.h \
                $(MODDIRI)/Math/IOptions.h \
                $(MODDIRI)/Math/GenAlgoOptions.h \
                $(MODDIRI)/Math/Integrator.h \
                $(MODDIRI)/Math/VirtualIntegrator.h \
                $(MODDIRI)/Math/AllIntegrationTypes.h \
                $(MODDIRI)/Math/AdaptiveIntegratorMultiDim.h \
                $(MODDIRI)/Math/IntegratorMultiDim.h \
                $(MODDIRI)/Math/Factory.h \
                $(MODDIRI)/Math/FitMethodFunction.h \
                $(MODDIRI)/Math/GaussIntegrator.h \
                $(MODDIRI)/Math/GaussLegendreIntegrator.h \
                $(MODDIRI)/Math/RootFinder.h \
                $(MODDIRI)/Math/IRootFinderMethod.h \
                $(MODDIRI)/Math/RichardsonDerivator.h \
                $(MODDIRI)/Math/BrentMethods.h \
                $(MODDIRI)/Math/BrentMinimizer1D.h \
                $(MODDIRI)/Math/BrentRootFinder.h \
                $(MODDIRI)/Math/DistSampler.h \
                $(MODDIRI)/Math/DistSamplerOptions.h \
                $(MODDIRI)/Math/GoFTest.h \
                $(MODDIRI)/Math/ChebyshevPol.h \
                $(MODDIRI)/Math/SpecFuncMathCore.h \
                $(MODDIRI)/Math/DistFuncMathCore.h
MATHCOREDH3  := $(filter-out $(MODDIRI)/Fit/Chi2Grad%,$(wildcard $(MODDIRI)/Fit/*.h))
MATHCOREDH3  := $(filter-out $(MODDIRI)/Fit/LinkDef%,$(MATHCOREDH3))

MATHCOREH1   := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MATHCOREH2   := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
MATHCOREH3   := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.icc))
MATHCOREH4   := $(filter-out $(MODDIRI)/Fit/LinkDef%,$(wildcard $(MODDIRI)/Fit/*.h))
MATHCOREH    := $(MATHCOREH1) $(MATHCOREH2) $(MATHCOREH3) $(MATHCOREH4)

MATHCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHCOREO    := $(call stripsrc,$(MATHCORES:.cxx=.o))

MATHCOREDEP  := $(MATHCOREO:.o=.d) $(MATHCOREDO:.o=.d)

MATHCORELIB  := $(LPATH)/libMathCore.$(SOEXT)
MATHCOREMAP  := $(MATHCORELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.icc,include/%.icc,\
	$(patsubst $(MODDIRI)/%.h,include/%.h,$(MATHCOREH)))
ALLLIBS      += $(MATHCORELIB)
ALLMAPS      += $(MATHCOREMAP)

# include all dependency files
INCLUDEFILES += $(MATHCOREDEP)


##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                test-$(MODNAME)

include/Math/%.h: $(MATHCOREDIRI)/Math/%.h
		@(if [ ! -d "include/Math" ]; then    \
		   mkdir -p include/Math;             \
		fi)
		cp $< $@

include/Math/%.icc: $(MATHCOREDIRI)/Math/%.icc
		@(if [ ! -d "include/Math" ]; then    \
		   mkdir -p include/Math;             \
		fi)
		cp $< $@

include/Fit/%.h: $(MATHCOREDIRI)/Fit/%.h
		@(if [ ! -d "include/Fit" ]; then     \
		   mkdir -p include/Fit;              \
		fi)
		cp $< $@

include/%.h:    $(MATHCOREDIRI)/%.h
		cp $< $@

$(MATHCORELIB): $(MATHCOREO) $(MATHCOREDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathCore.$(SOEXT) $@     \
		   "$(MATHCOREO) $(MATHCOREDO)" \
		   "$(MATHCORELIBEXTRA)"

$(call pcmrule,MATHCORE)
	$(noop)

$(MATHCOREDS):  $(MATHCOREDH1) $(MATHCOREDH2) $(MATHCOREDH3) $(MATHCOREL0) $(MATHCORELS) $(ROOTCLINGEXE) $(call pcmdep,MATHCORE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,MATHCORE) -c -writeEmptyRootPCM -I$(ROOT_SRCDIR) $(MATHCOREDH1) $(MATHCOREDH2) $(MATHCOREDH3) $(MATHCOREL0)

$(MATHCOREMAP): $(MATHCOREDH1) $(MATHCOREDH2) $(MATHCOREDH3) $(MATHCOREL0) $(MATHCORELS) $(ROOTCLINGEXE) $(call pcmdep,MATHCORE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(MATHCOREDS) $(call dictModule,MATHCORE) -c -I$(ROOT_SRCDIR) $(MATHCOREDH1) $(MATHCOREDH2) $(MATHCOREDH3) $(MATHCOREL0)

all-$(MODNAME): $(MATHCORELIB)

clean-$(MODNAME):
		@rm -f $(MATHCOREO) $(MATHCOREDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MATHCOREDEP) $(MATHCOREDS) $(MATHCOREDH) \
		   $(MATHCORELIB) $(MATHCOREMAP)
		@rm -rf include/Math include/Fit
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@rm -rf $(MATHCOREDIRT)
else
		@cd $(MATHCOREDIRT) && $(MAKE) distclean ROOTCONFIG=../../../bin/root-config
		@cd $(MATHCOREDIRT)/fit && $(MAKE) distclean ROOTCONFIG=../../../../bin/root-config
endif

distclean::     distclean-$(MODNAME)

test-$(MODNAME): all-$(MODNAME)
ifneq ($(ROOT_OBJDIR),$(ROOT_SRCDIR))
		@$(INSTALL) $(MATHCOREDIR)/test $(MATHCOREDIRT)
endif
		@cd $(MATHCOREDIRT) && $(MAKE) ROOTCONFIG=../../../bin/root-config
		@cd $(MATHCOREDIRT)/fit && $(MAKE) ROOTCONFIG=../../../../bin/root-config

##### extra rules ######
$(MATHCOREO): CXXFLAGS += -DUSE_ROOT_ERROR
$(MATHCOREDO): CXXFLAGS += -DUSE_ROOT_ERROR 
# add optimization to G__Math compilation
# Optimize dictionary with stl containers.
$(MATHCOREDO1) : NOOPT = $(OPT)
$(MATHCOREDO2) : NOOPT = $(OPT)
$(MATHCOREDO3) : NOOPT = $(OPT)
