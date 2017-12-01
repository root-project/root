# Module.mk for utilities for core/backport.
# Copyright (c) 1995-2017 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2017-11-30

MODNAME        := backport
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

BACKPORTDIR    := $(MODDIR)
BACKPORTDIRI   := $(BACKPORTDIR)/inc

BACKPORTH       = $(wildcard $(MODDIRI)/*.h) \
                  $(wildcard $(MODDIRI)/ROOT/*.hxx)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%,include/%,$(BACKPORTH))

##### local rules #####
include/%.h:    $(BACKPORTDIRI)/%.h
		cp $< $@

include/%.hxx:	$(BACKPORTDIRI)/%.hxx
		mkdir -p include/ROOT
		cp $< $@
