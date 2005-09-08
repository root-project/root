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

GSLVERS      := gsl-1.5
GSLSRCS      := $(MODDIRS)/$(GSLVERS).tar.gz
GSLDIRS      := $(MODDIRS)/$(GSLVERS)
GSLDIRI      := -I$(MODDIRS)/$(GSLVERS)
GSLETAG      := $(MODDIRS)/headers.d

##### libgsl #####
ifeq ($(PLATFORM),win32)
GSLLIBA      := $(GSLDIRS)/.libs/libgsl.lib
GSLLIB       := $(LPATH)/libgsl.lib
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
GSLBLD        = "libgsl - Win32 Debug"
else
GSLBLD        = "libgsl - Win32 Release"
endif
else
GSLLIBA      := $(GSLDIRS)/.libs/libgsl.a
GSLLIB       := $(LPATH)/libgsl.a
endif
GSLDEP       := $(GSLLIB)
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
GSLDBG      = "--enable-gdb"
else
GSLDBG      =
endif

##### libMathMore #####
MATHMOREL    := $(MODDIRI)/MathMore/LinkDef.h
MATHMORELINC := $(MODDIRI)/MathMore/LinkDef_SpecFunc.h \
		$(MODDIRI)/MathMore/LinkDef_StatFunc.h \
		$(MODDIRI)/MathMore/LinkDef_RootFinding.h \
		$(MODDIRI)/MathMore/LinkDef_Func.h 
MATHMOREDS   := $(MODDIRS)/G__MathMore.cxx
MATHMOREDO   := $(MATHMOREDS:.cxx=.o)
MATHMOREDH   := $(MATHMOREDS:.cxx=.h)
MATHMOREDH1  := $(MODDIRI)/MathMore/ProbFunc.h \
		$(MODDIRI)/MathMore/ProbFuncInv.h \
		$(MODDIRI)/MathMore/SpecFunc.h \
		$(MODDIRI)/MathMore/IGenFunction.h \
		$(MODDIRI)/MathMore/IParamFunction.h \
		$(MODDIRI)/MathMore/ParamFunction.h \
		$(MODDIRI)/MathMore/WrappedFunction.h \
		$(MODDIRI)/MathMore/Polynomial.h \
		$(MODDIRI)/MathMore/Derivator.h \
		$(MODDIRI)/MathMore/Interpolator.h \
		$(MODDIRI)/MathMore/InterpolationTypes.h \
		$(MODDIRI)/MathMore/RootFinder.h \
		$(MODDIRI)/MathMore/GSLRootFinder.h \
		$(MODDIRI)/MathMore/GSLRootFinderDeriv.h \
		$(MODDIRI)/MathMore/RootFinderAlgorithms.h \
		$(MODDIRI)/MathMore/Integrator.h \
		$(MODDIRI)/MathMore/Chebyshev.h

MATHMOREH    := $(filter-out $(MODDIRI)/MathMore/LinkDef%,$(wildcard $(MODDIRI)/MathMore/*.h))
MATHMORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHMOREO    := $(MATHMORES:.cxx=.o)

MATHMOREDEP  := $(MATHMOREO:.o=.d) $(MATHMOREDO:.o=.d)

MATHMORELIB  := $(LPATH)/libMathMore.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/MathMore/%.h,include/MathMore/%.h,$(MATHMOREH))
ALLLIBS      += $(MATHMORELIB)

# include all dependency files
INCLUDEFILES += $(MATHMOREDEP)

##### local rules #####
include/MathMore/%.h: $(MATHMOREDIRI)/MathMore/%.h
		@(if [ ! -d "include/MathMore" ]; then     \
		   mkdir include/MathMore;                 \
		fi)
		cp $< $@

$(GSLLIB):      $(GSLLIBA)
		cp $< $@

$(GSLLIBA):     $(GSLSRCS)
ifeq ($(PLATFORM),win32)
		@(if [ -d $(GSLDIRS) ]; then \
			rm -rf $(GSLDIRS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(MATHMOREDIRS); \
		if [ ! -d $(GSLVERS) ]; then \
			gunzip -c $(GSLVERS).tar.gz | tar xf -; \
		fi; \
		cd $(GSLVERS); \
		GNUMAKE=$(MAKE) ./configure $(GSLDBG) CC=cl LD=cl CFLAGS="$(CFLAGS)" ;  \
		cd gsl; sed -e 's/ln -s/cp -p/' Makefile > MakefileNew; mv MakefileNew Makefile; cd ../; \
		$(MAKE)) \
# 		unset MAKEFLAGS; \
# 		nmake -nologo -f gsl.mak \
# 		CFG=$(GSLBLD))
else
		@(if [ -d $(GSLDIRS) ]; then \
			rm -rf $(GSLDIRS); \
		fi; \
		echo "*** Building $@..."; \
		cd $(MATHMOREDIRS); \
		if [ ! -d $(GSLVERS) ]; then \
			gunzip -c $(GSLVERS).tar.gz | tar xf -; \
		fi; \
		cd $(GSLVERS); \
		ACC=$(CC); \
		ACFLAGS="-O"; \
		if [ "$(CC)" = "icc" ]; then \
			ACC="icc"; \
		fi; \
		if [ "$(ARCH)" = "sgicc64" ]; then \
			ACC="gcc -mabi=64"; \
		fi; \
		if [ "$(ARCH)" = "hpuxia64acc" ]; then \
			ACC="cc +DD64 -Ae"; \
		fi; \
		if [ "$(ARCH)" = "linuxppc64gcc" ]; then \
			ACC="gcc -m64"; \
		fi; \
		if [ "$(ARCH)" = "linuxx8664gcc" ]; then \
			ACC="gcc -m64"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure $(GSLDBG);  \
		$(MAKE))
endif

$(MATHMORELIB): $(GSLDEP) $(MATHMOREO) $(MATHMOREDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathMore.$(SOEXT) $@     \
		   "$(MATHMOREO) $(MATHMOREDO)"             \
		   "$(MATHMORELIBEXTRA) $(GSLLIB)"

$(MATHMOREDS):  $(MATHMOREDH1) $(MATHMOREL) $(MATHMORELINC) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MATHMOREDH1) $(MATHMOREL)

$(MATHMOREDO):  $(MATHMOREDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-mathmore:   $(MATHMORELIB)

map-mathmore:   $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(MATHMORELIB) \
		   -d $(MATHMORELIBDEP) -c $(MATHMOREL) $(MATHMORELINC)

map::           map-mathmore

clean-mathmore:
		@rm -f $(MATHMOREO) $(MATHMOREDO)
ifeq ($(PLATFORM),win32)
		-@(if [ -d $(GSLDIRS) ]; then \
			cd $(GSLDIRS); \
			unset MAKEFLAGS; \
			nmake -nologo -f gsl.mak clean \
			CFG=$(GSLBLD); \
		fi)
else
		-@(if [ -d $(GSLDIRS) ]; then \
			cd $(GSLDIRS); \
			$(MAKE) clean; \
		fi)
endif

clean::         clean-mathmore

distclean-mathmore: clean-mathmore
		@rm -f $(MATHMOREDEP) $(MATHMOREDS) $(MATHMOREDH) $(MATHMORELIB)
		@rm -rf $(GSLLIB) $(GSLDIRS)
		@rm -rf include/MathMore

distclean::     distclean-mathmore

##### extra rules ######
$(MATHMOREO): %.o: %.cxx
	$(CXX) $(OPT) $(CXXFLAGS) $(GSLDIRI) -o $@ -c $<
