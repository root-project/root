# Module.mk for unix module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := unix
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UNIXDIR      := $(MODDIR)
UNIXDIRS     := $(UNIXDIR)/src
UNIXDIRI     := $(UNIXDIR)/inc

##### libUnix (part of libCore) #####
UNIXL        := $(MODDIRI)/LinkDef.h

UNIXH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
UNIXS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
UNIXO        := $(call stripsrc,$(UNIXS:.cxx=.o))

UNIXDEP      := $(UNIXO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(UNIXH))

# include all dependency files
INCLUDEFILES += $(UNIXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(UNIXDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(UNIXO)

clean-$(MODNAME):
		@rm -f $(UNIXO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(UNIXDEP)

distclean::     distclean-$(MODNAME)
