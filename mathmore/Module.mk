# Module.mk for mathmore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := mathmore
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
MATHMORELINC := $(MODDIRI)/Math/LinkDef_SpecFunc.h \
		$(MODDIRI)/Math/LinkDef_StatFunc.h \
		$(MODDIRI)/Math/LinkDef_RootFinding.h \
		$(MODDIRI)/Math/LinkDef_Func.h
MATHMOREDS   := $(MODDIRS)/G__MathMore.cxx
MATHMOREDO   := $(MATHMOREDS:.cxx=.o)
MATHMOREDH   := $(MATHMOREDS:.cxx=.h)
MATHMOREDH1  := $(MODDIRI)/Math/DistFuncMathMore.h \
		$(MODDIRI)/Math/SpecFuncMathMore.h \
		$(MODDIRI)/Math/Polynomial.h \
		$(MODDIRI)/Math/Derivator.h \
		$(MODDIRI)/Math/Interpolator.h \
		$(MODDIRI)/Math/InterpolationTypes.h \
		$(MODDIRI)/Math/RootFinder.h \
		$(MODDIRI)/Math/GSLRootFinder.h \
		$(MODDIRI)/Math/GSLRootFinderDeriv.h \
		$(MODDIRI)/Math/RootFinderAlgorithms.h \
		$(MODDIRI)/Math/Integrator.h \
		$(MODDIRI)/Math/Minimizer1D.h \
		$(MODDIRI)/Math/Chebyshev.h  \
		$(MODDIRI)/Math/Random.h \
		$(MODDIRI)/Math/GSLRndmEngines.h \
		$(MODDIRI)/Math/KelvinFunctions.h \
		$(MODDIRS)/GSLError.h

MATHMOREH    := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
MATHMORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHMOREO    := $(MATHMORES:.cxx=.o)

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
include/Math/%.h: $(MATHMOREDIRI)/Math/%.h
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

$(MATHMORELIB): $(MATHMOREO) $(MATHMOREDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathMore.$(SOEXT) $@     \
		   "$(MATHMOREO) $(MATHMOREDO)"             \
		   "$(MATHMORELIBEXTRA) $(GSLLIBDIR) $(GSLLIBS)"

$(MATHMOREDS):  $(MATHMOREDH1) $(MATHMOREL) $(MATHMORELINC) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP)  -f  $@ -c $(MATHMOREDH1)  $(MATHMOREL)

$(MATHMOREMAP): $(RLIBMAP) $(MAKEFILEDEP) $(MATHMOREL) $(MATHMORELINC)
		$(RLIBMAP) -o $(MATHMOREMAP) -l $(MATHMORELIB) \
		   -d $(MATHMORELIBDEPM) -c $(MATHMOREL) $(MATHMORELINC)

all-mathmore:   $(MATHMORELIB) $(MATHMOREMAP)

clean-mathmore:
		@rm -f $(MATHMOREO) $(MATHMOREDO)

clean::         clean-mathmore

distclean-mathmore: clean-mathmore
		@rm -f $(MATHMOREDEP) $(MATHMOREDS) $(MATHMOREDH) \
		   $(MATHMORELIB) $(MATHMOREMAP)
		@rm -rf include/Math

distclean::     distclean-mathmore

##### extra rules ######
$(MATHMOREO): CXXFLAGS += $(GSLFLAGS)
$(MATHMOREDO): CXXFLAGS += $(GSLFLAGS)
