# Module.mk for mathcore module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODDIR       := mathcore
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MATHCOREDIR  := $(MODDIR)
MATHCOREDIRS := $(MATHCOREDIR)/src
MATHCOREDIRI := $(MATHCOREDIR)/inc

##### libMathCore #####
MATHCOREL1   := $(MODDIRI)/LinkDef.h
MATHCOREL2   := $(MODDIRI)/Math/LinkDef.h
MATHCORELINC := $(MODDIRI)/Math/LinkDef_Func.h
MATHCOREDS1  := $(MODDIRS)/G__Math.cxx
MATHCOREDS2  := $(MODDIRS)/G__MathCore.cxx
MATHCOREDO1  := $(MATHCOREDS1:.cxx=.o)
MATHCOREDO2  := $(MATHCOREDS2:.cxx=.o)
MATHCOREL    := $(MATHCOREL1) $(MATHCOREL2)
MATHCOREDS   := $(MATHCOREDS1) $(MATHCOREDS2)
MATHCOREDO   := $(MATHCOREDO1) $(MATHCOREDO2)
MATHCOREDH   := $(MATHCOREDS:.cxx=.h)

MATHCOREDH1  :=  $(MODDIRI)/TComplex.h \
                 $(MODDIRI)/TMath.h \
                 $(MODDIRI)/TRandom.h \
                 $(MODDIRI)/TRandom1.h \
                 $(MODDIRI)/TRandom2.h \
                 $(MODDIRI)/TRandom3.h
MATHCOREDH2  :=  $(MODDIRI)/Math/SpecFuncMathCore.h \
                 $(MODDIRI)/Math/DistFuncMathCore.h \
                 $(MODDIRI)/Math/IParamFunction.h \
                 $(MODDIRI)/Math/IFunction.h \
                 $(MODDIRI)/Math/Functor.h \
                 $(MODDIRI)/Math/Minimizer.h \
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
                 $(MODDIRI)/Math/BrentRootFinder.h

MATHCOREH1   := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MATHCOREH2   := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
MATHCOREH    := $(MATHCOREH1) $(MATHCOREH2)

MATHCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHCOREO    := $(MATHCORES:.cxx=.o)

MATHCOREDEP  := $(MATHCOREO:.o=.d) $(MATHCOREDO:.o=.d)

MATHCORELIB  := $(LPATH)/libMathCore.$(SOEXT)
MATHCOREMAP  := $(MATHCORELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MATHCOREH))
ALLLIBS      += $(MATHCORELIB)
ALLMAPS      += $(MATHCOREMAP)

# include all dependency files
INCLUDEFILES += $(MATHCOREDEP)

#all link def used for libMathCore
ALLMATHCOREL := $(MATHCOREL) $(MATHCORELINC) $(FITL)

##### local rules #####
include/Math/%.h: $(MATHCOREDIRI)/Math/%.h
		@(if [ ! -d "include/Math" ]; then   \
		   mkdir -p include/Math;       \
		fi)
		cp $< $@

include/%.h:    $(MATHCOREDIRI)/%.h
		cp $< $@

$(MATHCORELIB): $(MATHCOREO) $(MATHCOREDO) $(FITO) $(FITDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathCore.$(SOEXT) $@     \
		   "$(MATHCOREO) $(MATHCOREDO) $(FITO) $(FITDO)" \
		   "$(MATHCORELIBEXTRA)"

$(MATHCOREDS1): $(MATHCOREDH1) $(MATHCOREL1) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		@echo "for files $(MATHCOREDH1)"
		$(ROOTCINTTMP) -f $@ -c $(MATHCOREDH1) $(MATHCOREL1)

$(MATHCOREDS2): $(MATHCOREDH2) $(MATHCOREL2) $(MATHCORELINC) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		@echo "for files $(MATHCOREDH2)"
		$(ROOTCINTTMP) -f $@ -c $(MATHCOREDH2) $(MATHCOREL2)

$(MATHCOREMAP): $(RLIBMAP) $(MAKEFILEDEP) $(ALLMATHCOREL)
		$(RLIBMAP) -o $(MATHCOREMAP) -l $(MATHCORELIB) \
		   -d $(MATHCORELIBDEPM) -c $(ALLMATHCOREL)

all-mathcore:   $(MATHCORELIB) $(MATHCOREMAP)

clean-mathcore:
		@rm -f $(MATHCOREO) $(MATHCOREDO)

clean::         clean-mathcore

distclean-mathcore: clean-mathcore
		@rm -f $(MATHCOREDEP) $(MATHCOREDS) $(MATHCOREDH) \
		   $(MATHCORELIB) $(MATHCOREMAP)
		@rm -rf include/Math

distclean::     distclean-mathcore

test-mathcore:	all-mathcore
		@cd $(MATHCOREDIR)/test; make

##### extra rules ######
$(MATHCOREO): CXXFLAGS += -DUSE_ROOT_ERROR
$(MATHCOREDO): CXXFLAGS += -DUSE_ROOT_ERROR
