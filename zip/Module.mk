# Module.mk for zip module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := zip
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

ZIPDIR       := $(MODDIR)
ZIPDIRS      := $(ZIPDIR)/src
ZIPDIRI      := $(ZIPDIR)/inc

##### libZip (part of libCore) #####
ZIPH         := $(wildcard $(MODDIRI)/*.h)
ZIPS         := $(wildcard $(MODDIRS)/*.c)
ZIPO         := $(ZIPS:.c=.o)

ZIPDEP       := $(ZIPO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(ZIPH))

# include all dependency files
INCLUDEFILES += $(ZIPDEP)

##### local rules #####
include/%.h:    $(ZIPDIRI)/%.h
		cp $< $@

all-zip:        $(ZIPO)

clean-zip:
		@rm -f $(ZIPO)

clean::         clean-zip

distclean-zip:  clean-zip
		@rm -f $(ZIPDEP)

distclean::     distclean-zip
