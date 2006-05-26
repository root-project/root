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

##### libgsl #####
ifeq ($(BUILDGSL),yes)
GSLDIRI      := -I$(MODDIRS)/$(GSLVERS)
ifeq ($(PLATFORM),win32)
GSLLIBA      := $(GSLDIRS)/libgsl.lib
#GSLLIB       := $(LPATH)/libgsl.lib
ifeq (yes,$(WINRTDEBUG))
GSLBLD        = "libgsl - Win32 Debug"
else
GSLBLD        = "libgsl - Win32 Release"
endif
else
GSLLIBA      := $(GSLDIRS)/.libs/libgsl.a
#GSLLIB       := $(LPATH)/libgsl.a
GSLOPT       := $(OPT)
endif
GSLDEP       := $(GSLLIBA)
ifeq (debug,$(findstring debug,$(ROOTBUILD)))
GSLDBG        = "--enable-gdb"
else
GSLDBG        =
endif
else
GSLDIRI      := $(GSLFLAGS)
GSLLIBA      := $(GSLLIBS)
GSLDEP       :=
GSLDBG       :=
endif

##### libMathMore #####
MATHMOREL    := $(MODDIRI)/Math/LinkDef.h
MATHMORELINC := $(MODDIRI)/Math/LinkDef_SpecFunc.h \
		$(MODDIRI)/Math/LinkDef_StatFunc.h \
		$(MODDIRI)/Math/LinkDef_RootFinding.h \
		$(MODDIRI)/Math/LinkDef_Func.h
MATHMOREDS   := $(MODDIRS)/G__MathMore.cxx
MATHMOREDO   := $(MATHMOREDS:.cxx=.o)
MATHMOREDH   := $(MATHMOREDS:.cxx=.h)
MATHMOREDH1  := $(MODDIRI)/Math/ProbFuncMathMore.h \
		$(MODDIRI)/Math/ProbFuncInv.h \
		$(MODDIRI)/Math/SpecFuncMathMore.h \
		$(MODDIRI)/Math/IGenFunction.h \
		$(MODDIRI)/Math/IParamFunction.h \
		$(MODDIRI)/Math/ParamFunction.h \
		$(MODDIRI)/Math/WrappedFunction.h \
		$(MODDIRI)/Math/Polynomial.h \
		$(MODDIRI)/Math/Derivator.h \
		$(MODDIRI)/Math/Interpolator.h \
		$(MODDIRI)/Math/InterpolationTypes.h \
		$(MODDIRI)/Math/RootFinder.h \
		$(MODDIRI)/Math/GSLRootFinder.h \
		$(MODDIRI)/Math/GSLRootFinderDeriv.h \
		$(MODDIRI)/Math/RootFinderAlgorithms.h \
		$(MODDIRI)/Math/Integrator.h \
		$(MODDIRI)/Math/Chebyshev.h  \
		$(MODDIRI)/Math/Random.h \
		$(MODDIRI)/Math/GSLRndmEngines.h

MATHMOREH    := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
MATHMORES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MATHMOREO    := $(MATHMORES:.cxx=.o)

MATHMOREDEP  := $(MATHMOREO:.o=.d) $(MATHMOREDO:.o=.d)

MATHMORELIB  := $(LPATH)/libMathMore.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(MATHMOREH))
ALLLIBS      += $(MATHMORELIB)

# include all dependency files
INCLUDEFILES += $(MATHMOREDEP)

##### local rules #####
include/Math/%.h: $(MATHMOREDIRI)/Math/%.h
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

#$(GSLLIB):      $(GSLLIBA)
#		cp $< $@

ifeq ($(BUILDGSL),yes)
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
		cp ./*.h ./gsl; \
		cp ./gsl_version.h.win32 ./gsl_version.h; \
		cp ./config.h.win32 ./config.h; \
		cp ./*/*.h ./gsl; \
		unset MAKEFLAGS; \
		nmake -f Makefile.msc CFG=$(GSLBLD))
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
			ACC="gcc"; \
			ACFLAGS="-m64"; \
		fi; \
		GNUMAKE=$(MAKE) ./configure CC="$$ACC" \
		CFLAGS="$$ACFLAGS $(GSLOPT)" $(GSLDBG); \
		if [ "$(MACOSX_CPU)" = "i386" ]; then \
			sed '/DARWIN_IEEE_INTERFACE/d' config.status > _c.s; \
			rm -f config.status config.h; \
			cp _c.s config.status; \
		fi; \
		$(MAKE))
endif
endif

$(MATHMORELIB): $(GSLDEP) $(MATHMOREO) $(MATHMOREDO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"  \
		   "$(SOFLAGS)" libMathMore.$(SOEXT) $@     \
		   "$(MATHMOREO) $(MATHMOREDO)"             \
		   "$(MATHMORELIBEXTRA) $(GSLLIBA)"

$(MATHMOREDS):  $(MATHMOREDH1) $(MATHMOREL) $(MATHMORELINC) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MATHMOREDH1) $(MATHMOREL)

all-mathmore:   $(MATHMORELIB)

map-mathmore:   $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(MATHMORELIB) \
		   -d $(MATHMORELIBDEP) -c $(MATHMOREL) $(MATHMORELINC)

map::           map-mathmore

clean-mathmore:
		@rm -f $(MATHMOREO) $(MATHMOREDO)
ifeq ($(BUILDGSL),yes)
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
endif

clean::         clean-mathmore

distclean-mathmore: clean-mathmore
		@rm -f $(MATHMOREDEP) $(MATHMOREDS) $(MATHMOREDH) $(MATHMORELIB)
		@mv $(GSLSRCS) $(MATHMOREDIRS)/-$(GSLVERS).tar.gz
		@rm -rf $(MATHMOREDIRS)/gsl-*
		@mv $(MATHMOREDIRS)/-$(GSLVERS).tar.gz $(GSLSRCS)
		@rm -rf include/Math

distclean::     distclean-mathmore

##### extra rules ######
$(MATHMOREO): $(GSLDEP)
$(MATHMOREO): CXXFLAGS += $(GSLDIRI)
