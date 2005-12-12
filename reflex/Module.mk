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
REFLEXAH     := $(wildcard $(MODDIRI)/Reflex/*.h)
REFLEXBH     := $(wildcard $(MODDIRI)/Reflex/Builder/*.h)
REFLEXH      := $(REFLEXAH) $(REFLEXBH)
REFLEXS      := $(wildcard $(MODDIRS)/*.cxx)
REFLEXO      := $(REFLEXS:.cxx=.o)

REFLEXDEP    := $(REFLEXO:.o=.d)

REFLEXLIB    := $(LPATH)/libReflex.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Reflex/%.h,include/Reflex/%.h,$(REFLEXH))
ALLLIBS      += $(REFLEXLIB)

# include all dependency files
INCLUDEFILES += $(REFLEXDEP)

# genreflex
GRFLXSD := $(REFLEXDIR)/python/genreflex
GRFLXDD := lib/python/genreflex

GCCXMLPATHPY := $(GRFLXDD)/gccxmlpath.py

GRFLXS   := $(wildcard $(GRFLXSD)/*.py)
GRFLXPY  := $(patsubst $(GRFLXSD)/%.py,$(GRFLXDD)/%.py,$(GRFLXS))
GRFLXPY  += $(GCCXMLPATHPY)
ifneq ($(BUILDPYTHON),no)
GRFLXPYC := $(subst .py,.pyc,$(GRFLXPY))
endif

ifeq ($(PLATFORM),win32)
GENREFLEX = bin/genreflex.bat
GNRFLX_L1 = "" #"@echo off"
GNRFLX_L2 = "" #"python  %~d0%~p0\..\lib\python\genreflex\genreflex.py %*"
GENRFLXRC = bin/genreflex-rootcint.bat
GRFLXRC_L1 = "" #"@echo off"
GRFLXRC_L2 = "" #"python %~d0%~p0\..\lib\python\genreflex\genreflex-rootcint.py %*"
else
GENREFLEX = bin/genreflex
GNRFLX_L1 = "\#!/bin/csh -f"
GNRFLX_L2 = 'python $$0:h/../lib/python/genreflex/genreflex.py $$*'
GENRFLXRC = bin/genreflex-rootcint
GRFLXRC_L1 = "\#!/bin/csh -f"
GRFLXRC_L2 = 'python $$0:h/../lib/python/genreflex/genreflex-rootcint.py $$*'
endif

# test suite
CPPUNITI   = -I$(CPPUNIT)/include
CPPUNITLL  = -L$(CPPUNIT)/lib -lcppunit
REFLEXLL   = -Llib -lReflex

RFLX_GENREFLEXX = ../../bin/genreflex

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

##### local rules #####
include/Reflex/%.h: $(REFLEXDIRI)/Reflex/%.h
		@(if [ ! -d "include/Reflex" ]; then    \
		   mkdir -p include/Reflex/Builder;     \
		fi)
		cp $< $@

.PRECIOUS: $(GRFLXPY)

$(GCCXMLPATHPY):
		@(if [ ! -d "lib/python/genreflex" ]; then \
		  mkdir -p lib/python/genreflex; fi )
		@echo "gccxmlpath = '$(GCCXML)'" > $(GCCXMLPATHPY);

$(GRFLXDD)/%.py: $(GRFLXSD)/%.py $(GCCXMLPATHPY)
		@(if [ ! -d "lib/python/genreflex" ]; then \
		  mkdir -p lib/python/genreflex; fi )
		cp $< $@

$(GRFLXDD)/%.pyc: $(GRFLXDD)/%.py
		@python -c 'import py_compile; py_compile.compile( "$<" )'

$(GENREFLEX): $(GRFLXPYC)
		@echo $(GNRFLX_L1) > $(GENREFLEX)
		@echo $(GNRFLX_L2) >> $(GENREFLEX)
ifneq ($(PLATFORM),win32)
		@chmod a+x $(GENREFLEX)
endif

$(GENRFLXRC) : $(GRFLXPYC)
		@echo $(GRFLXRC_L1) > $(GENRFLXRC)
		@echo $(GRFLXRC_L2) >> $(GENRFLXRC)
ifneq ($(PLATFORM),win32)
		@chmod a+x $(GENRFLXRC)
endif

$(REFLEXLIB): $(GENREFLEX) $(GENRFLXRC) $(REFLEXO) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libReflex.$(SOEXT) $@ "$(REFLEXO)" \
		"$(REFLEXLIBEXTRA)"

all-reflex:     $(REFLEXLIB)

map-reflex:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(REFLEXLIB) \
		-d $(REFLEXLIBDEP) -c $(REFLEXL)

map::           map-reflex

clean-genreflex:
		@rm -f bin/genreflex*
		@rm -fr lib/python/genreflex

clean-check-reflex:
		@rm -f $(RFLX_TESTLIBS) $(RFLX_TESTLIBO) $(RFLX_UNITTESTO) $(RFLX_UNITTESTX)

clean-reflex: clean-genreflex clean-check-reflex
		@rm -f $(REFLEXO)

clean::         clean-reflex

distclean-reflex: clean-reflex
		@rm -f $(REFLEXDEP) $(REFLEXLIB)
		@rm -rf include/Reflex

distclean::     distclean-reflex

# test suite

check-reflex: $(REFLEXLIB) $(RFLX_TESTLIB) $(RFLX_UNITTESTX)
		@if [ ! -e lib/libcppunit.$(SOEXT) ]; then ln -s $(CPPUNIT)/lib/libcppunit.$(SOEXT) lib/libcppunit.$(SOEXT); fi
		$(RFLX_TESTD)/test_Reflex_generate
		$(RFLX_TESTD)/test_Reflex_simple1
		$(RFLX_TESTD)/test_Reflex_simple2
		$(RFLX_TESTD)/test_Reflex_unit
		$(RFLX_TESTD)/test_ReflexBuilder_unit

lib/libtest_%Rflx.$(SOEXT) : $(RFLX_TESTD)/%_rflx.o
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $@ $@ $< $(REFLEXLL)

%_rflx.o : %_rflx.cpp
		$(CXX) $(OPT) $(CXXFLAGS) -c $< -o $@

$(RFLX_TESTLIBS1) :
		cd $(RFLX_TESTD); $(RFLX_GENREFLEXX) ../../include/Reflex/Reflex.h -s testDict1/selection.xml -I../../include

$(RFLX_TESTLIBS2) :
		cd $(RFLX_TESTD); $(RFLX_GENREFLEXX) testDict2/Class2Dict.h -s testDict2/selection.xml -I../../include

$(RFLX_UNITTESTO) : %.o : %.cxx
		$(CXX) $(OPT) $(CXXFLAGS) $(CPPUNITI) -Ireflex -c $< -o $@

$(RFLX_UNITTESTX) : % : %.o
		$(LD) $(LDFLAGS) -o $@ $< $(CPPUNITLL) $(REFLEXLL) -ldl

