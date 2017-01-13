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

MATHCOREDICTH :=  TComplex.h \
                TMath.h \
                TRandom.h \
                TRandom1.h \
                TRandom2.h \
                TRandom3.h \
                TStatistic.h \
                TKDTree.h \
                TKDTreeBinning.h \
                Math/Random.h \
                Math/RandomFunctions.h \
                Math/TRandomEngine.h \
                Math/MersenneTwisterEngine.h \
                Math/MixMaxEngine.h \
                Math/KDTree.h \
                Math/TDataPoint.h \
                Math/TDataPointN.h \
                Math/IParamFunction.h \
                Math/IFunction.h \
                Math/ParamFunctor.h \
                Math/Functor.h \
                Math/Minimizer.h \
                Math/MinimizerOptions.h \
                Math/MinimTransformFunction.h \
                Math/MinimTransformVariable.h \
                Math/BasicMinimizer.h \
                Math/IntegratorOptions.h \
                Math/IOptions.h \
                Math/GenAlgoOptions.h \
                Math/Integrator.h \
                Math/VirtualIntegrator.h \
                Math/AllIntegrationTypes.h \
                Math/AdaptiveIntegratorMultiDim.h \
                Math/IntegratorMultiDim.h \
                Math/Factory.h \
                Math/FitMethodFunction.h \
                Math/GaussIntegrator.h \
                Math/GaussLegendreIntegrator.h \
                Math/RootFinder.h \
                Math/IRootFinderMethod.h \
                Math/RichardsonDerivator.h \
                Math/BrentMethods.h \
                Math/BrentMinimizer1D.h \
                Math/BrentRootFinder.h \
                Math/DistSampler.h \
                Math/DistSamplerOptions.h \
                Math/GoFTest.h \
                Math/ChebyshevPol.h \
                Math/SpecFuncMathCore.h \
                Math/DistFuncMathCore.h \
		$(patsubst $(MODDIRI)/%,%,$(filter-out $(MODDIRI)/Fit/LinkDef%,$(filter-out $(MODDIRI)/Fit/Chi2Grad%,$(wildcard $(MODDIRI)/Fit/*.h))))

MATHCOREMH1   := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MATHCOREMH2   := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
MATHCOREMH3   := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.icc))
MATHCOREMH4   := $(filter-out $(MODDIRI)/Fit/LinkDef%,$(wildcard $(MODDIRI)/Fit/*.h))
MATHCOREMH    := $(MATHCOREMH1) $(MATHCOREMH2) $(MATHCOREMH3) $(MATHCOREMH4)

MATHCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHCORECS   := $(wildcard $(MODDIRS)/*.c)
MATHCOREO    := $(call stripsrc,$(MATHCORES:.cxx=.o) $(MATHCORECS:.c=.o))

MATHCOREDEP  := $(MATHCOREO:.o=.d) $(MATHCOREDO:.o=.d)

MATHCORELIB  := $(LPATH)/libMathCore.$(SOEXT)
MATHCOREMAP  := $(MATHCORELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
MATHCOREMH_REL := $(patsubst $(MODDIRI)/%,include/%,$(MATHCOREMH))
ALLHDRS      += $(MATHCOREMH_REL)
ALLLIBS      += $(MATHCORELIB)
ALLMAPS      += $(MATHCOREMAP)
ifeq ($(CXXMODULES),yes)
  MATHCOREMH_NOICC_REL := $(filter-out $(patsubst $(MODDIRI)/%,include/%,$(MATHCOREMH3)), $(MATHCOREMH_REL))
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(MATHCOREMH_NOICC_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Math_Core { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(MATHCORELIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

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

$(MATHCOREDS):  $(add-prefix include/,$(MATHCOREDICTH)) $(MATHCOREL0) $(MATHCORELS) $(ROOTCLINGEXE) $(call pcmdep,MATHCORE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,MATHCORE) -c -writeEmptyRootPCM -I$(ROOT_SRCDIR) $(MATHCOREDICTH) $(MATHCOREL0)

$(MATHCOREMAP): $(add-prefix include/,$(MATHCOREDICTH)) $(MATHCOREL0) $(MATHCORELS) $(ROOTCLINGEXE) $(call pcmdep,MATHCORE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(MATHCOREDS) $(call dictModule,MATHCORE) -c -I$(ROOT_SRCDIR) $(MATHCOREDHICTH) $(MATHCOREL0)

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
