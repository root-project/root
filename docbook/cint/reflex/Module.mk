# Module.mk for reflex module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := reflex
MODDIR       := $(ROOT_SRCDIR)/cint/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

# see also ModuleVars.mk

##### libReflex #####
REFLEXL      := $(MODDIRI)/LinkDef.h
REFLEXDS     := $(call stripsrc,$(MODDIRS)/G__Reflex.cxx)
REFLEXDO     := $(REFLEXDS:.cxx=.o)
REFLEXDH     := $(REFLEXDS:.cxx=.h)

REFLEXAH     := $(wildcard $(MODDIRI)/Reflex/*.h)
REFLEXBH     := $(wildcard $(MODDIRI)/Reflex/Builder/*.h)
REFLEXIH     := $(wildcard $(MODDIRI)/Reflex/internal/*.h)
REFLEXH      := $(REFLEXAH) $(REFLEXBH) $(REFLEXIH)
REFLEXAPIH   := $(filter-out $(MODDIRI)/Reflex/Builder/ReflexBuilder.h,\
	        $(filter-out $(MODDIRI)/Reflex/Reflex.h,\
	        $(filter-out $(MODDIRI)/Reflex/SharedLibrary.h,\
	        $(filter-out $(MODDIRI)/Reflex/DictionaryGenerator.h,\
		$(REFLEXAH) $(REFLEXBH)))))

REFLEXDEP    := $(REFLEXO:.o=.d) $(REFLEXDO:.o=.d)

REFLEXDICTLIB:= $(LPATH)/libReflexDict.$(SOEXT)
REFLEXDICTMAP:= $(REFLEXDICTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Reflex/%.h,include/Reflex/%.h,$(REFLEXH))
ALLLIBS      += $(REFLEXLIB) $(REFLEXDICTLIB)
ALLMAPS      += $(REFLEXDICTMAP)

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

RFLX_LIBDIR = $(LIBDIR)

ifeq ($(PLATFORM),win32)
# test suite
RFLX_CPPUNITI   = "$(shell cygpath -w '$(CPPUNIT)/include')"
RFLX_CPPUNITLL  = "$(shell cygpath -w '$(CPPUNIT)/lib/cppunit.lib')"
else
# test suite
RFLX_CPPUNITI   = $(CPPUNIT)/include
RFLX_CPPUNITLL  = -L$(CPPUNIT)/lib -lcppunit
endif

RFLX_TESTD      = $(call stripsrc,$(REFLEXDIR)/test)
RFLX_TESTDL     = $(RFLX_TESTD)/lib
RFLX_TESTLIBD1  = $(RFLX_TESTD)/testDict1
RFLX_TESTLIBD2  = $(RFLX_TESTD)/testDict2
RFLX_TESTLIBS1  = $(RFLX_TESTD)/Reflex_rflx.cpp
RFLX_TESTLIBS2  = $(RFLX_TESTD)/Class2Dict_rflx.cpp
RFLX_TESTLIBS   = $(RFLX_TESTLIBS1) $(RFLX_TESTLIBS2)
RFLX_TESTLIBO   = $(subst .cpp,.o,$(RFLX_TESTLIBS))
RFLX_TESTLIB    = $(subst $(RFLX_TESTD)/,$(RFLX_TESTDL)/libtest_,$(subst _rflx.o,Rflx.$(SOEXT),$(RFLX_TESTLIBO)))
.PRECIOUS: $(RFLX_TESTLIB) $(RFLX_TESTLIBO)

RFLX_UNITTESTS = $(RFLX_TESTD)/test_Reflex_generate.cxx    \
                 $(RFLX_TESTD)/test_ReflexBuilder_unit.cxx \
                 $(RFLX_TESTD)/test_Reflex_unit.cxx        \
                 $(RFLX_TESTD)/test_Reflex_simple1.cxx     \
                 $(RFLX_TESTD)/test_Reflex_simple2.cxx
RFLX_UNITTESTO = $(subst .cxx,.o,$(RFLX_UNITTESTS))
RFLX_UNITTESTX = $(subst .cxx,,$(RFLX_UNITTESTS))

RFLX_GENMAPS   = $(REFLEXDIRS)/genmap/genmap.cxx
RFLX_GENMAPO   = $(call stripsrc,$(RFLX_GENMAPS:.cxx=.o))
RFLX_GENMAPDEP = $(RFLX_GENMAPO:.o=.d)

# include all dependency files
INCLUDEFILES += $(RFLX_GENMAPDEP)

ALLEXECS += $(RFLX_GENMAPX)

POSTBIN  += $(RFLX_GRFLXPYC) $(RFLX_GRFLXPY)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME) \
                clean-check-$(MODNAME) check-$(MODNAME) clean-genreflex

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

$(RFLX_GENMAPO) : CXXFLAGS += -I$(REFLEXDIRS)/genmap

$(RFLX_GENMAPX) : $(RFLX_GENMAPO) $(REFLEXLIB)
	$(LD) $(LDFLAGS) -o $@ $(RFLX_GENMAPO) $(RFLX_REFLEXLL)

$(REFLEXLIB): $(REFLEXO)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libReflex.$(SOEXT) $@ "$(REFLEXO)" \
		"$(REFLEXLIBEXTRA)"

$(REFLEXDICTLIB): $(REFLEXDO) $(ORDER_) $(MAINLIBS) $(REFLEXLIB)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libReflexDict.$(SOEXT) $@ "$(REFLEXDO)" \
		"$(REFLEXDICTLIBEXTRA)"

$(REFLEXDS): $(REFLEXAPIH) $(REFLEXL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -p -Icint/reflex/inc $(REFLEXAPIH) $(REFLEXL)

$(REFLEXDICTMAP): $(RLIBMAP) $(MAKEFILEDEP) $(REFLEXL)
		$(RLIBMAP) -o $@ -l $(REFLEXDICTLIB) \
		   -d $(REFLEXDICTLIBDEPM) -c $(REFLEXL)

all-$(MODNAME): $(REFLEXLIB) $(REFLEXDICTLIB) $(REFLEXDICTMAP) $(RFLX_GRFLXPYC) $(RFLX_GRFLXPY)

clean-genreflex:
		@rm -rf lib/python/genreflex

clean-check-$(MODNAME):
		@rm -f $(RFLX_TESTLIB) $(RFLX_TESTLIBS) $(RFLX_TESTLIBO) $(RFLX_UNITTESTO) $(RFLX_UNITTESTX) \
		       $(RFLX_TESTD)/test_Reflex_generate.testout

clean-$(MODNAME): clean-genreflex clean-check-$(MODNAME)
		@rm -f $(RFLX_GENMAPX)
		@rm -f $(REFLEXO) $(REFLEXDO) $(RFLX_GENMAPO) $(RFLX_UNITTESTO) $(RFLX_TESTLIBO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(REFLEXDEP) $(REFLEXDS) $(REFLEXDH) $(REFLEXLIB) \
		   $(REFLEXDICTLIB) $(REFLEXDICTMAP) $(RFLX_GENMAPDEP)  $(RFLX_TESTLIB)
		@rm -rf include/Reflex lib/python

distclean::     distclean-$(MODNAME)

# test suite

ifeq ($(PLATFORM),win32)
RFLX_EXPORTTESTPATH := "$(RFLX_TESTDL):`pwd`/bin:$(CPPUNIT)/lib:$(PATH)"
else
RFLX_EXPORTTESTPATH := export LD_LIBRARY_PATH=$(RFLX_TESTDL):`pwd`/lib:$(CPPUNIT)/lib:$$LD_LIBRARY_PATH
endif

check-reflex-run%: $(RFLX_TESTD)/% $(REFLEXLIB) $(RFLX_TESTLIB)
		$(RFLX_EXPORTTESTPATH); $(RFLX_TESTD)/$*

check-reflex: $(addprefix check-reflex-run,$(notdir $(RFLX_UNITTESTX))) check-reflex-diff-generate

$(RFLX_TESTDL)/libtest_%Rflx.$(SOEXT) : $(RFLX_TESTD)/%_rflx.o
		@mkdir -p $(dir $@)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" "$(SOFLAGS)" $(notdir $@) $@ $< $(RFLX_REFLEXLL)

%_rflx.o : %_rflx.cpp
		$(CXX) $(OPT) $(CXXFLAGS) -c $< $(CXXOUT)$@

$(RFLX_TESTLIBS1) : $(REFLEXDIRI)/Reflex/Reflex.h $(RFLX_TESTLIBD1)/selection.xml
		cd $(RFLX_TESTD); ../../../bin/genreflex testDict1/Reflex.h -s testDict1/selection.xml -I../../../include

$(RFLX_TESTLIBS2) : $(RFLX_TESTLIBD2)/Class2Dict.h $(RFLX_TESTLIBD2)/selection.xml $(wildcard $(RFLX_TESTLIBD2)/*.h)
		cd $(RFLX_TESTD); ../../../bin/genreflex testDict2/Class2Dict.h -s testDict2/selection.xml -I../../../include --iocomments

$(RFLX_UNITTESTO) : $(RFLX_TESTD)/test_Reflex%.o : $(RFLX_TESTD)/test_Reflex%.cxx $(RFLX_TESTD)/CppUnit_testdriver.cpp
		$(CXX) $(OPT) $(CXXFLAGS) -I$(RFLX_TESTD) -Icint/reflex -I$(RFLX_CPPUNITI) -c $< $(CXXOUT)$@

$(RFLX_UNITTESTX) : $(RFLX_TESTD)/test_Reflex% : $(RFLX_TESTD)/test_Reflex%.o
		$(LD) $(LDFLAGS) -o $@ $< $(RFLX_CPPUNITLL) $(RFLX_REFLEXLL)

.PHONY: check-reflex-diff-generate
$(RFLX_TESTD)/test_Reflex_generate.testout: $(RFLX_TESTD)/test_Reflex_generate $(REFLEXLIB) $(RFLX_TESTLIB)
	$(RFLX_EXPORTTESTPATH); $(RFLX_TESTD)/test_Reflex_generate

check-reflex-diff-generate: $(RFLX_TESTD)/test_Reflex_generate.testout $(RFLX_TESTD)/test_Reflex_generate.testref
	@diff -u $(RFLX_TESTD)/test_Reflex_generate.testref $(RFLX_TESTD)/test_Reflex_generate.testout
