# Module.mk for clib module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := clib
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CLIBDIR      := $(MODDIR)
CLIBDIRS     := $(CLIBDIR)/src
CLIBDIRI     := $(CLIBDIR)/inc

##### libClib (part of libCore) #####
CLIBL        := $(MODDIRI)/LinkDef.h

CLIBH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CLIBHH       := $(CLIBDIRI)/strlcpy.h $(CLIBDIRI)/snprintf.h 
CLIBS1       := $(wildcard $(MODDIRS)/*.c)
CLIBS2       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

CLIBO        := $(call stripsrc,$(CLIBS1:.c=.o) $(CLIBS2:.cxx=.o))
SNPRINTFO    := $(call stripsrc,$(CLIBDIRS)/snprintf.o)
STRLCPYO     := $(call stripsrc,$(CLIBDIRS)/strlcpy.o $(CLIBDIRS)/strlcat.o)

CLIBDEP      := $(CLIBO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CLIBH))

# include all dependency files
INCLUDEFILES += $(CLIBDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CLIBDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(CLIBO)

clean-$(MODNAME):
		@rm -f $(CLIBO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLIBDEP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
