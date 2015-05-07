# Module.mk for mathmore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := mathmore
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATHMOREDIR  := $(MODDIR)
MATHMOREDIRS := $(MATHMOREDIR)/src
MATHMOREDIRI := $(MATHMOREDIR)/inc

###pre-compiled GSL DLL require Mathmore to be compiled with -DGSL_DLL
#ifeq ($(PLATFORM),win32)
#GSLFLAGS += "-DGSL_DLL"
#endif

##### libMathMore #####
MATHMOREL    := $(MODDIRI)/Math/LinkDef.h
MATHMORELINC := $(MODDIRI)/Math/LinkDef_Func.h \
                $(MODDIRI)/Math/LinkDef_RootFinding.h

MATHMOREDS   := $(call stripsrc,$(MODDIRS)/G__MathMore.cxx)
MATHMOREDO   := $(MATHMOREDS:.cxx=.o)
MATHMOREDH   := $(MATHMOREDS:.cxx=.h)
MATHMOREDH1  := $(MODDIRI)/Math/DistFuncMathMore.h \
                $(MODDIRI)/Math/SpecFuncMathMore.h \
                $(MODDIRI)/Math/PdfFuncMathMore.h \
                $(MODDIRI)/Math/Polynomial.h \
                $(MODDIRI)/Math/Derivator.h \
                $(MODDIRI)/Math/Interpolator.h \
                $(MODDIRI)/Math/InterpolationTypes.h \
                $(MODDIRI)/Math/GSLRootFinder.h \
                $(MODDIRI)/Math/GSLRootFinderDeriv.h \
                $(MODDIRI)/Math/RootFinderAlgorithms.h \
                $(MODDIRI)/Math/GSLIntegrator.h \
                $(MODDIRI)/Math/GSLMCIntegrator.h \
                $(MODDIRI)/Math/MCParameters.h \
                $(MODDIRI)/Math/GSLMinimizer1D.h \
                $(MODDIRI)/Math/ChebyshevApprox.h  \
                $(MODDIRI)/Math/Random.h \
                $(MODDIRI)/Math/GSLRndmEngines.h \
                $(MODDIRI)/Math/QuasiRandom.h \
                $(MODDIRI)/Math/GSLQuasiRandom.h \
                $(MODDIRI)/Math/KelvinFunctions.h \
                $(MODDIRI)/Math/GSLMinimizer.h \
                $(MODDIRI)/Math/GSLNLSMinimizer.h \
                $(MODDIRI)/Math/GSLSimAnMinimizer.h \
                $(MODDIRI)/Math/GSLMultiRootFinder.h \
                $(MODDIRI)/Math/Vavilov.h \
                $(MODDIRI)/Math/VavilovAccurate.h \
                $(MODDIRI)/Math/VavilovAccuratePdf.h \
                $(MODDIRI)/Math/VavilovAccurateCdf.h \
                $(MODDIRI)/Math/VavilovAccurateQuantile.h \
                $(MODDIRI)/Math/VavilovFast.h 

#                $(MODDIRS)/GSLError.h

MATHMOREH    := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
MATHMORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHMOREO    := $(call stripsrc,$(MATHMORES:.cxx=.o))

MATHMOREDEP  := $(MATHMOREO:.o=.d) $(MATHMOREDO:.o=.d)

MATHMORELIB  := $(LPATH)/libMathMore.$(SOEXT)
MATHMOREMAP  := $(MATHMORELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(MATHMOREH))
ALLLIBS      += $(MATHMORELIB)
ALLMAPS      += $(MATHMOREMAP)

# include all dependency files
INCLUDEFILES += $(MATHMOREDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/Math/%.h: $(MATHMOREDIRI)/Math/%.h
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

$(MATHMORELIB): $(MATHMOREO) $(MATHMOREDO) $(ORDER_) $(MAINLIBS) $(MATHMORELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathMore.$(SOEXT) $@     \
		   "$(MATHMOREO) $(MATHMOREDO)"             \
		   "$(MATHMORELIBEXTRA) $(GSLLIBDIR) $(GSLLIBS)"

$(call pcmrule,MATHMORE)
	$(noop)

$(MATHMOREDS):  $(MATHMOREDH1) $(MATHMOREL) $(MATHMORELINC) $(ROOTCLINGEXE) $(call pcmdep,MATHMORE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@  $(call dictModule,MATHMORE) -c -writeEmptyRootPCM $(ROOT_SRCDIR:%=-I%) $(GSLFLAGS) $(MATHMOREDH1) $(MATHMOREL)

$(MATHMOREMAP): $(MATHMOREDH1) $(MATHMOREL) $(MATHMORELINC) $(ROOTCLINGEXE) $(call pcmdep,MATHMORE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(MATHMOREDS)  $(call dictModule,MATHMORE) -c $(ROOT_SRCDIR:%=-I%) $(GSLFLAGS) $(MATHMOREDH1) $(MATHMOREL)

all-$(MODNAME): $(MATHMORELIB)

clean-$(MODNAME):
		@rm -f $(MATHMOREO) $(MATHMOREDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MATHMOREDEP) $(MATHMOREDS) $(MATHMOREDH) \
		   $(MATHMORELIB) $(MATHMOREMAP)
		@rm -rf include/Math

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(MATHMOREO): CXXFLAGS += $(GSLFLAGS)  -DUSE_ROOT_ERROR
$(MATHMOREDO): CXXFLAGS += $(ROOT_SRCDIR:%=-I%) $(GSLFLAGS) -DUSE_ROOT_ERROR

# Optimize dictionary with stl containers.
$(MATHMOREDO): NOOPT = $(OPT)
