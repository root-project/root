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
GRFLXPYC := $(subst .py,.pyc,$(GRFLXPY))

ifeq ($(PLATFORM),win32)
GENREFLEX = bin/genreflex.bat
GNRFLX_L1 = "" #"@echo off"
GNRFLX_L2 = "" #"python  %~d0%~p0\..\lib\python\genreflex\genreflex.py %*"
else
GENREFLEX = bin/genreflex
GNRFLX_L1 = "\#!/bin/csh -f"
GNRFLX_L2 = 'python $$0:h/../lib/python/genreflex/genreflex.py $$*'
endif

# test suite
CPPUNITI   = -I$(CPPUNIT)/include
CPPUNITLL  = -L$(CPPUNIT)/lib -lcppunit
REFLEXLL   = -Llib -lReflex

GENREFLEXX = ../../bin/genreflex

TESTD      = $(REFLEXDIR)/test
TESTLIBD1  = $(TESTD)/testDict1
TESTLIBD2  = $(TESTD)/testDict2
TESTLIBS1  = $(TESTD)/Reflex_rflx.cpp
TESTLIBS2  = $(TESTD)/Class2Dict_rflx.cpp
TESTLIBS   = $(TESTLIBS1) $(TESTLIBS2)
TESTLIBO   = $(subst .cpp,.o,$(TESTLIBS))
TESTLIB    = $(subst $(TESTD)/,lib/libtest_,$(subst _rflx.o,Rflx.$(SOEXT),$(TESTLIBO)))

UNITTESTS = $(TESTD)/test_Reflex_generate.cxx    \
            $(TESTD)/test_ReflexBuilder_unit.cxx \
            $(TESTD)/test_Reflex_unit.cxx        \
            $(TESTD)/test_Reflex_simple1.cxx     \
            $(TESTD)/test_Reflex_simple2.cxx 
UNITTESTO = $(subst .cxx,.o,$(UNITTESTS))
UNITTESTX = $(subst .cxx,,$(UNITTESTS))

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

$(REFLEXLIB): $(GENREFLEX) $(REFLEXO)
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
		@rm -f $(TESTLIBS) $(TESTLIBO) $(UNITTESTO) $(UNITTESTX)

clean-reflex: clean-genreflex clean-check-reflex
		@rm -f $(REFLEXO) 

clean::         clean-reflex

distclean-reflex: clean-reflex
		@rm -f $(REFLEXDEP) $(REFLEXLIB)
		@rm -rf include/Reflex

distclean::     distclean-reflex

# test suite

check-reflex: $(REFLEXLIB) $(TESTLIB) $(UNITTESTX)
		@if [ ! -e lib/libcppunit.$(SOEXT) ]; then ln -s $(CPPUNIT)/lib/libcppunit.$(SOEXT) lib/libcppunit.$(SOEXT); fi
		$(TESTD)/test_Reflex_generate
		$(TESTD)/test_Reflex_simple1
		$(TESTD)/test_Reflex_simple2
		$(TESTD)/test_Reflex_unit
		$(TESTD)/test_ReflexBuilder_unit

lib/libtest_%Rflx.$(SOEXT) : $(TESTD)/%_rflx.o
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $@ $@ $< $(REFLEXLL)

%_rflx.o : %_rflx.cpp
		$(CXX) $(OPT) $(CXXFLAGS) -c $< -o $@

$(TESTLIBS1) :
		cd $(TESTD); $(GENREFLEXX) ../../include/Reflex/Reflex.h -s testDict1/selection.xml -I../../include

$(TESTLIBS2) :
		cd $(TESTD); $(GENREFLEXX) testDict2/Class2Dict.h -s testDict2/selection.xml -I../../include

$(UNITTESTO) : %.o : %.cxx
		$(CXX) $(OPT) $(CXXFLAGS) $(CPPUNITI) -Ireflex -c $< -o $@ 

$(UNITTESTX) : % : %.o
		$(LD) $(LDFLAGS) -o $@ $< $(CPPUNITLL) $(REFLEXLL)

