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
MATHCOREL    := $(MODDIRI)/Math/LinkDef.h
MATHCORELINC := $(MODDIRI)/Math/LinkDef_Func.h 

MATHCOREDS   := $(MODDIRS)/G__MathCore.cxx

MATHCOREDO   := $(MATHCOREDS:.cxx=.o)

MATHCOREDH   := $(MATHCOREDS:.cxx=.h)

MATHCOREDH1  :=   \
                 $(MODDIRI)/Math/SpecFuncMathCore.h \
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




MATHCOREH   := $(filter-out $(MODDIRI)/Math/LinkDef%, $(wildcard $(MODDIRI)/Math/*.h))

MATHCORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHCOREO    := $(MATHCORES:.cxx=.o)

MATHCOREDEP  := $(MATHCOREO:.o=.d)  $(MATHCOREDO:.o=.d) 

MATHCORELIB  := $(LPATH)/libMathCore.$(SOEXT)
MATHCOREMAP  := $(MATHCORELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(MATHCOREH))
ALLLIBS      += $(MATHCORELIB)
ALLMAPS      += $(MATHCOREMAP)

# include all dependency files
INCLUDEFILES += $(MATHCOREDEP)

#all link def used for libMathCore
ALLMATHCOREL := $(MATHCOREL) $(MATHCORELINC) $(MATHL) $(FITL)


##### local rules #####
include/Math/%.h: $(MATHCOREDIRI)/Math/%.h
		@(if [ ! -d "include/Math" ]; then   \
		   mkdir -p include/Math;       \
		fi)
		cp $< $@
# build lib mathcore: use also obj  from math and fit directory 
$(MATHCORELIB): $(MATHCOREO) $(MATHCOREDO)  $(MATHO) $(MATHDO) $(FITO) $(FITDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathCore.$(SOEXT) $@     \
		   "$(MATHCOREO) $(MATHCOREDO)  $(MATHO) $(MATHDO) $(FITO) $(FITDO)"    \
		   "$(MATHCORELIBEXTRA)"

$(MATHCOREDS):  $(MATHCOREDH1) $(MATHCOREL) $(MATHCORELINC) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		@echo "for files $(MATHCOREDH1)"
		$(ROOTCINTTMP) -f $@ -c $(MATHCOREDH1) $(MATHCOREL)
#		genreflex $(MATHCOREDIRS)/MathCoreDict.h  --selection_file=$(MATHCOREDIRS)/selection_MathCore.xml -o $(MATHCOREDIRS)/G__MathCore.cxx -I$(MATHCOREDIRI)


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

