# Module.mk for ios module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Timur Pocheptsov, 17/7/2011

MODNAME      := ios
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

IOSDIR       := $(MODDIR)
IOSDIRS      := $(IOSDIR)/src
IOSDIRI      := $(IOSDIR)/inc

##### libIOS (part of libRoot.a) #####
IOSH         := $(wildcard $(MODDIRI)/*.h)
IOSS         := $(wildcard $(MODDIRS)/*.cxx)
IOSO         := $(call stripsrc,$(IOSS:.cxx=.o))

IOSDEP       := $(IOSO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(IOSH))

# include all dependency files
INCLUDEFILES += $(IOSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(IOSDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(IOSO)

clean-$(MODNAME):
		@rm -f $(IOSO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(IOSDEP)

distclean::     distclean-$(MODNAME)
