# Module.mk for reflex module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := reflex
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

REFLEXDIR    := $(MODDIR)
REFLEXDIRS   := $(REFLEXDIR)/src
REFLEXDIRI   := $(REFLEXDIR)/inc

##### libReflex #####
REFLEXL      := $(MODDIRI)/LinkDef.h
REFLEXDS     := $(MODDIRS)/G__Reflex.cxx
REFLEXDO     := $(REFLEXDS:.cxx=.o)
REFLEXDH     := $(REFLEXDS:.cxx=.h)

REFLEXAH     := $(wildcard $(MODDIRI)/Reflex/*.h)
REFLEXBH     := $(wildcard $(MODDIRI)/Reflex/Builder/*.h)
REFLEXIH     := $(wildcard $(MODDIRI)/Reflex/internal/*.h)
REFLEXH      := $(REFLEXAH) $(REFLEXBH) $(REFLEXIH)
REFLEXAPIH   := $(filter-out $(MODDIRI)/Reflex/Builder/ReflexBuilder.h,\
	        $(filter-out $(MODDIRI)/Reflex/Reflex.h,\
	        $(filter-out $(MODDIRI)/Reflex/SharedLibrary.h,\
		$(REFLEXAH) $(REFLEXBH))))
REFLEXS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
REFLEXO      := $(REFLEXS:.cxx=.o)

REFLEXDEP    := $(REFLEXO:.o=.d) $(REFLEXDO:.o=.d)

REFLEXLIB    := $(LPATH)/libReflex.$(SOEXT)
REFLEXDICTLIB:= $(LPATH)/libReflexDict.$(SOEXT)
REFLEXMAP    := $(REFLEXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Reflex/%.h,include/Reflex/%.h,$(REFLEXH))
ALLLIBS      += $(REFLEXLIB) $(REFLEXDICTLIB)
ALLMAPS      += $(REFLEXMAP)

# include all dependency files
INCLUDEFILES += $(REFLEXDEP)

# genreflex
RFLX_GRFLXSD := $(REFLEXDIR)/python/genreflex
RFLX_GRFLXDD := lib/python/genreflex

RFLX_GCCXMLPATHPY := $(RFLX_GRFLXDD)/gccxmlpath.py

RFLX_GRFLXS   := $(wildcard $(RFLX_GRFLXSD)/*.py)
RFLX_GRFLXPY  := $(patsubst $(RFLX_GRFLXSD)/%.py,$(RFLX_GRFLXDD)/%.py,$(RFLX_GRFLXS))
RFLX_GRFLXPY  += $(RFLX_GCCXMLPATHPY)
ifneq ($(BUILDPYTHON),no)
RFLX_GRFLXPYC := $(subst .py,.pyc,$(RFLX_GRFLXPY))
endif

ifeq ($(PLATFORM),win32)
RFLX_LIBDIR = %~d0%~p0\..\lib
else
RFLX_LIBDIR = `dirname $$0`/../lib
endif

ifeq ($(PLATFORM),win32)
RFLX_GENREFLEX = bin/genreflex.bat
RFLX_GNRFLX_L1 = "@echo off"
RFLX_GNRFLX_L2 = "python  $(RFLX_LIBDIR)\python\genreflex\genreflex.py %*"
RFLX_GENRFLXRC = bin/genreflex-rootcint.bat
RFLX_GRFLXRC_L1 = "@echo off"
RFLX_GRFLXRC_L2 = "python $(RFLX_LIBDIR)\python\genreflex\genreflex-rootcint.py %*"
# test suite
RFLX_CPPUNITI   = "$(shell cygpath -w '$(CPPUNIT)/include')"
RFLX_CPPUNITLL  = "$(shell cygpath -w '$(CPPUNIT)/lib/cppunit.lib')"
RFLX_REFLEXLL   = lib/libReflex.lib
else
RFLX_GENREFLEX = bin/genreflex
RFLX_GNRFLX_L1 = "\#!/bin/sh"
RFLX_GNRFLX_L2 = 'python $(RFLX_LIBDIR)/python/genreflex/genreflex.py "$$@"'
RFLX_GENRFLXRC = bin/genreflex-rootcint
RFLX_GRFLXRC_L1 = "\#!/bin/sh"
RFLX_GRFLXRC_L2 = 'python $(RFLX_LIBDIR)/python/genreflex/genreflex-rootcint.py "$$@"'
# test suite
RFLX_CPPUNITI   = $(CPPUNIT)/include
RFLX_CPPUNITLL  = -L$(CPPUNIT)/lib -lcppunit
RFLX_REFLEXLL   = -Llib -lReflex
ifneq ($(PLATFORM),fbsd)
ifneq ($(PLATFORM),obsd)
RFLX_REFLEXLL   += -ldl
endif
endif
endif

ifeq ($(PLATFORM),solaris)
RFLX_REFLEXLL   += -ldemangle
endif

RFLX_GENREFLEX_CMD = python ../../lib/python/genreflex/genreflex.py

RFLX_TESTD      = $(REFLEXDIR)/test
RFLX_TESTLIBD1  = $(RFLX_TESTD)/testDict1
RFLX_TESTLIBD2  = $(RFLX_TESTD)/testDict2
RFLX_TESTLIBS1  = $(RFLX_TESTD)/Reflex_rflx.cpp
RFLX_TESTLIBS2  = $(RFLX_TESTD)/Class2Dict_rflx.cpp
RFLX_TESTLIBS   = $(RFLX_TESTLIBS1) $(RFLX_TESTLIBS2)
RFLX_TESTLIBO   = $(subst .cpp,.o,$(RFLX_TESTLIBS))
RFLX_TESTLIB    = $(subst $(RFLX_TESTD)/,lib/libtest_,$(subst _rflx.o,Rflx.$(SOEXT),$(RFLX_TESTLIBO)))

RFLX_UNITTESTS = $(RFLX_TESTD)/test_Reflex_generate.cxx    \
                 $(RFLX_TESTD)/test_ReflexBuilder_unit.cxx \
                 $(RFLX_TESTD)/test_Reflex_unit.cxx        \
                 $(RFLX_TESTD)/test_Reflex_simple1.cxx     \
                 $(RFLX_TESTD)/test_Reflex_simple2.cxx
RFLX_UNITTESTO = $(subst .cxx,.o,$(RFLX_UNITTESTS))
RFLX_UNITTESTX = $(subst .cxx,,$(RFLX_UNITTESTS))

RFLX_GENMAPS   = $(REFLEXDIRS)/genmap/genmap.cxx
RFLX_GENMAPO   = $(RFLX_GENMAPS:.cxx=.o)
RFLX_GENMAPX   = bin/genmap$(EXEEXT)

ALLEXECS += $(RFLX_GENREFLEX) $(RFLX_GENRFLXRC) $(RFLX_GENMAPX)

##### local rules #####
include/Reflex/%.h: $(REFLEXDIRI)/Reflex/%.h
		@(if [ ! -d "include/Reflex" ]; then          \
		   mkdir -p include/Reflex;                   \
		fi)
		@(if [ ! -d "include/Reflex/Builder" ]; then  \
		   mkdir -p include/Reflex/Builder;           \
		fi)
		@(if [ ! -d "include/Reflex/internal" ]; then \
		   mkdir -p include/Reflex/internal;          \
		fi)
		cp $< $@

.PRECIOUS: $(RFLX_GRFLXPY)

$(RFLX_GCCXMLPATHPY): config/Makefile.config
		@(if [ ! -d "lib/python/genreflex" ]; then \
		  mkdir -p lib/python/genreflex; fi )
		@echo "gccxmlpath = '$(GCCXML)'" > $(RFLX_GCCXMLPATHPY);

$(RFLX_GRFLXDD)/%.py: $(RFLX_GRFLXSD)/%.py $(RFLX_GCCXMLPATHPY)
		@(if [ ! -d "lib/python/genreflex" ]; then \
		  mkdir -p lib/python/genreflex; fi )
		cp $< $@

$(RFLX_GRFLXDD)/%.pyc: $(RFLX_GRFLXDD)/%.py
		@python -c 'import py_compile; py_compile.compile( "$<" )'

$(RFLX_GENREFLEX): $(RFLX_GRFLXPYC)
		@echo $(RFLX_GNRFLX_L1) > $(RFLX_GENREFLEX)
		@echo $(RFLX_GNRFLX_L2) >> $(RFLX_GENREFLEX)
ifneq ($(PLATFORM),win32)
		@chmod a+x $(RFLX_GENREFLEX)
endif

$(RFLX_GENRFLXRC) : $(RFLX_GRFLXPYC)
		@echo $(RFLX_GRFLXRC_L1) > $(RFLX_GENRFLXRC)
		@echo $(RFLX_GRFLXRC_L2) >> $(RFLX_GENRFLXRC)
ifneq ($(PLATFORM),win32)
		@chmod a+x $(RFLX_GENRFLXRC)
endif

$(RFLX_GENMAPO) : $(RFLX_GENMAPS)
	$(CXX) $(OPT) $(CXXFLAGS) -Iinclude -I$(REFLEXDIRS)/genmap -c $< $(CXXOUT)$@

$(RFLX_GENMAPX) : $(RFLX_GENMAPO) $(REFLEXLIB)
	$(LD) $(LDFLAGS) -o $@ $(RFLX_GENMAPO) $(RFLX_REFLEXLL)

$(REFLEXLIB): $(RFLX_GENREFLEX) $(RFLX_GENRFLXRC) $(REFLEXO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libReflex.$(SOEXT) $@ "$(REFLEXO)" \
		"$(REFLEXLIBEXTRA)"

$(REFLEXDICTLIB): $(REFLEXDO) $(ORDER_) $(MAINLIBS) $(REFLEXLIB)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libReflexDict.$(SOEXT) $@ "$(REFLEXDO)" \
		"$(RFLX_REFLEXLL) $(REFLEXLIBEXTRA)"

$(REFLEXDS): $(REFLEXAPIH) $(REFLEXL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -p $(REFLEXAPIH) $(REFLEXL)

$(REFLEXMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(REFLEXL)
		$(RLIBMAP) -o $(REFLEXMAP) -l $(REFLEXLIB) \
		   -d $(REFLEXLIBDEPM) -c $(REFLEXL)

all-reflex:     $(REFLEXLIB) $(REFLEXDICTLIB) $(REFLEXMAP)

clean-genreflex:
		@rm -f bin/genreflex*
		@rm -rf lib/python/genreflex

clean-check-reflex:
		@rm -f $(RFLX_TESTLIBS) $(RFLX_TESTLIBO) $(RFLX_UNITTESTO) $(RFLX_UNITTESTX)

clean-reflex: clean-genreflex clean-check-reflex
		@rm -f $(RFLX_GENMAPX)
		@rm -f $(REFLEXO) $(REFLEXDO)

clean::         clean-reflex

distclean-reflex: clean-reflex
		@rm -f $(REFLEXDEP) $(REFLEXLIB) $(REFLEXMAP)
		@rm -rf include/Reflex lib/python

distclean::     distclean-reflex

# test suite

check-reflex: $(REFLEXLIB) $(RFLX_TESTLIB) $(RFLX_UNITTESTX)
ifeq ($(PLATFORM),win32)
		@export PATH="`pwd`/bin:$(CPPUNIT)/lib:$(PATH)"; \
		$(RFLX_TESTD)/test_Reflex_generate; \
		$(RFLX_TESTD)/test_Reflex_simple1; \
		$(RFLX_TESTD)/test_Reflex_simple2; \
		$(RFLX_TESTD)/test_Reflex_unit; \
		$(RFLX_TESTD)/test_ReflexBuilder_unit
else
		@export LD_LIBRARY_PATH=`pwd`/lib:$(CPPUNIT)/lib; \
		$(RFLX_TESTD)/test_Reflex_generate; \
		$(RFLX_TESTD)/test_Reflex_simple1; \
		$(RFLX_TESTD)/test_Reflex_simple2; \
		$(RFLX_TESTD)/test_Reflex_unit; \
		$(RFLX_TESTD)/test_ReflexBuilder_unit
endif

lib/libtest_%Rflx.$(SOEXT) : $(RFLX_TESTD)/%_rflx.o
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ $< $(RFLX_REFLEXLL)

%_rflx.o : %_rflx.cpp
		$(CXX) $(OPT) $(CXXFLAGS) -c $< $(CXXOUT)$@

$(RFLX_TESTLIBS1) : $(REFLEXDIRI)/Reflex/Reflex.h $(RFLX_TESTLIBD1)/selection.xml
		cd $(RFLX_TESTD); $(RFLX_GENREFLEX_CMD) testDict1/Reflex.h -s testDict1/selection.xml -I../../include

$(RFLX_TESTLIBS2) : $(RFLX_TESTLIBD2)/Class2Dict.h $(RFLX_TESTLIBD2)/selection.xml $(wildcard $(RFLX_TESTLIBD2)/*.h)
		cd $(RFLX_TESTD); $(RFLX_GENREFLEX_CMD) testDict2/Class2Dict.h -s testDict2/selection.xml -I../../include

$(RFLX_UNITTESTO) : $(RFLX_TESTD)/test_Reflex%.o : $(RFLX_TESTD)/test_Reflex%.cxx
		$(CXX) $(OPT) $(CXXFLAGS) -I$(RFLX_CPPUNITI) -Ireflex -c $< $(CXXOUT)$@

$(RFLX_UNITTESTX) : $(RFLX_TESTD)/test_Reflex% : $(RFLX_TESTD)/test_Reflex%.o
		$(LD) $(LDFLAGS) -o $@ $< $(RFLX_CPPUNITLL) $(RFLX_REFLEXLL)

