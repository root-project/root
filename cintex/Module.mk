# Module.mk for reflex module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := cintex
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CINTEXDIR    := $(MODDIR)
CINTEXDIRS   := $(CINTEXDIR)/src
CINTEXDIRI   := $(CINTEXDIR)/inc

##### libCintex #####
CINTEXH      := $(wildcard $(MODDIRI)/Cintex/*.h)
CINTEXS      := $(wildcard $(MODDIRS)/*.cxx)
CINTEXO      := $(CINTEXS:.cxx=.o)

CINTEXDEP    := $(CINTEXO:.o=.d)

CINTEXLIB    := $(LPATH)/libCintex.$(SOEXT)

CINTEXPYS    := $(wildcard $(MODDIR)/python/*.py)
ifeq ($(PLATFORM),win32)
CINTEXPY     := $(subst $(MODDIR)/python,bin,$(CINTEXPYS))
bin/%.py: $(MODDIR)/python/%.py; cp $< $@
else
CINTEXPY     := $(subst $(MODDIR)/python,$(LPATH),$(CINTEXPYS))
$(LPATH)/%.py: $(MODDIR)/python/%.py; cp $< $@
endif
CINTEXPYC    := $(CINTEXPY:.py=.pyc)
CINTEXPYO    := $(CINTEXPY:.py=.pyo)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/Cintex/%.h,include/Cintex/%.h,$(CINTEXH))
ALLLIBS      += $(CINTEXLIB)

# include all dependency files
INCLUDEFILES += $(CINTEXDEP)

##### local rules #####
include/Cintex/%.h: $(CINTEXDIRI)/Cintex/%.h
		@(if [ ! -d "include/Cintex" ]; then    \
		   mkdir -p include/Cintex;             \
		fi)
		cp $< $@

%.pyc: %.py;    python -c 'import py_compile; py_compile.compile( "$<" )'
%.pyo: %.py;    python -O -c 'import py_compile; py_compile.compile( "$<" )'

$(CINTEXLIB):   $(CINTEXO) $(MAINLIBS) $(CINTEXLIBDEP) $(CINTEXPY) $(CINTEXPYC) $(CINTEXPYO)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)"      \
		"$(SOFLAGS)" libCintex.$(SOEXT) $@ "$(CINTEXO)" \
		"$(CINTEXLIBEXTRA)"

all-cintex:     $(CINTEXLIB)

map-cintex:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(CINTEXLIB) \
		-d $(CINTEXLIBDEP) -c $(CINTEXL)

map::           map-cintex

clean-cintex:
		@rm -f $(CINTEXO)

clean::         clean-cintex

distclean-cintex: clean-cintex
		@rm -f $(CINTEXDEP) $(CINTEXLIB) $(CINTEXPY) $(CINTEXPYC) $(CINTEXPYO)
		@rm -rf include/Cintex

distclean::     distclean-cintex


