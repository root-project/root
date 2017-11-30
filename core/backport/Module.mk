# Module.mk for utilities for core/backport.
# Copyright (c) 1995-2017 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2017-11-30

MODNAME        := backport
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

FOUNDATIONDIR   := $(MODDIR)
FOUNDATIONDIRS  := $(FOUNDATIONDIR)/src
FOUNDATIONDIRI  := $(FOUNDATIONDIR)/inc
FOUNDATIONDIRR  := $(FOUNDATIONDIR)/res

BACKPORTTH       += $(MODDIRI)/libcpp_string_view.h
BACKPORTTH       += $(MODDIRI)/RWrap_libcpp_string_view.h
BACKPORTTH       += $(MODDIRI)/ROOT/span.hxx
BACKPORTTH       += $(MODDIRI)/ROOT/memory.hxx
BACKPORTTH       += $(MODDIRI)/ROOT/tuple.hxx

FOUNDATIONO     := $(call stripsrc,$(FOUNDATIONS:.cxx=.o))

FOUNDATIONL     := $(MODDIRI)/LinkDef.h

FOUNDATIONDEP   := $(FOUNDATIONO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%,include/%,$(FOUNDATIONH) $(FOUNDATIONTH))

# include all dependency files
INCLUDEFILES += $(FOUNDATIONDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FOUNDATIONDIRI)/%.h
		cp $< $@

include/%.hxx:	$(FOUNDATIONDIRI)/%.hxx
		mkdir -p include/ROOT
		cp $< $@

all-$(MODNAME): $(FOUNDATIONO)

clean-$(MODNAME):
		@rm -f $(FOUNDATIONO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FOUNDATIONDEP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(FOUNDATIONO): CXXFLAGS += -I$(FOUNDATIONDIRR)
