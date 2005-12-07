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


GRFLXSD := $(REFLEXDIR)/python/genreflex
GRFLXDD := lib/python/genreflex

GCCXMLPATHPY := $(GRFLXDD)/gccxmlpath.py

GRFLXS   := $(wildcard $(GRFLXSD)/*.py)
GRFLXPY  := $(patsubst $(GRFLXSD)/%.py,$(GRFLXDD)/%.py,$(GRFLXS))
GRFLXPY  += $(GCCXMLPATHPY)
GRFLXPYC := $(subst .py,.pyc,$(GRFLXPY))

ifeq ($(PLATFORM),win32)
GENREFLEX = bin\genreflex.bat
GNRFLX_L1 = "" #"@echo off"
GNRFLX_L2 = "" #"python %ROOTSYS%/lib/python/genreflex/genrefle.py %*"
else
GENREFLEX = bin/genreflex
GNRFLX_L1 = "\#!/bin/csh -f"
GNRFLX_L2 = 'python $${ROOTSYS}/lib/python/genreflex/genreflex.py $$*'
endif



##### local rules #####
include/Reflex/%.h: $(REFLEXDIRI)/Reflex/%.h
		@(if [ ! -d "include/Reflex" ]; then    \
		   mkdir -p include/Reflex/Builder;     \
		fi)
		cp $< $@

.PRECIOUS: $(GRFLXPY)

$(GRFLXDD)/%.py: $(GRFLXSD)/%.py
		cp $< $@

$(GCCXMLPATHPY):
		@(if [ ! -d "lib/python/genreflex" ]; then \
		  mkdir -p lib/python/genreflex; fi )
		@echo "gccxmlpath = '$(GCCXML)'" > $(GCCXMLPATHPY);

$(GRFLXDD)/%.pyc: $(GCCXMLPATHPY) $(GRFLXDD)/%.py
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

clean-reflex: clean-genreflex
		@rm -f $(REFLEXO) 

clean::         clean-reflex

distclean-reflex: clean-reflex
		@rm -f $(REFLEXDEP) $(REFLEXLIB)
		@rm -rf include/Reflex

distclean::     distclean-reflex

# test suite

testDict1: 
		cd $(REFLEXDIR)/test/testDict1; python ../../python/genreflex/genreflex.py ../../inc/Reflex/Reflex.h -s selection.xml --gccxmlpath=$(GCCXML) -I../../inc
		$(CXX) -c $(CXXFLAGS) $(REFLEXDIR)/test/testDict1/Reflex_rflx.cpp
		$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" libtestDict1.$(SOEXT) Reflex_rflx.o -lReflex

check-reflex: testDict1 

#testDict2 test_Reflex_unit test_ReflexBuilder_unit test_Reflex_simple1 test_Reflex_simple2 test_Reflex_generate
