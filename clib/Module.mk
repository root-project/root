# Module.mk for clib module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := clib
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CLIBDIR      := $(MODDIR)
CLIBDIRS     := $(CLIBDIR)/src
CLIBDIRI     := $(CLIBDIR)/inc

##### libClib (part of libCore) #####
CLIBH        := $(wildcard $(MODDIRI)/*.h)
CLIBS        := $(wildcard $(MODDIRS)/*.c)
CLIBO        := $(CLIBS:.c=.o)

CLIBDEP      := $(CLIBO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CLIBH))

# include all dependency files
INCLUDEFILES += $(CLIBDEP)

##### local rules #####
include/%.h:    $(CLIBDIRI)/%.h
		cp $< $@

all-clib:       $(CLIBO)

clean-clib:
		@rm -f $(CLIBO)

clean::         clean-clib

distclean-clib: clean-clib
		@rm -f $(CLIBDEP)

distclean::     distclean-clib
