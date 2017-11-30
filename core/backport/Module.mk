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

BACKPORTTH       = $(wildcard $(MODDIRI)/*.h) \
                   $(wildcard $(MODDIRI)/ROOT/*.hxx)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%,include/%,$(FOUNDATIONH) $(FOUNDATIONTH))

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FOUNDATIONDIRI)/%.h
		cp $< $@

include/%.hxx:	$(FOUNDATIONDIRI)/%.hxx
		mkdir -p include/ROOT
		cp $< $@
