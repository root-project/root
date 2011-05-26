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
CLIBDS       := $(call stripsrc,$(MODDIRS)/G__Clib.cxx)
CLIBDO       := $(CLIBDS:.cxx=.o)
CLIBDH       := $(CLIBDS:.cxx=.h)

CLIBH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CLIBHH       := $(CLIBDIRI)/strlcpy.h $(CLIBDIRI)/snprintf.h 
CLIBS1       := $(wildcard $(MODDIRS)/*.c)
CLIBS2       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

CLIBO        := $(call stripsrc,$(CLIBS1:.c=.o) $(CLIBS2:.cxx=.o))
SNPRINTFO    := $(call stripsrc,$(CLIBDIRS)/snprintf.o)
STRLCPYO     := $(call stripsrc,$(CLIBDIRS)/strlcpy.o $(CLIBDIRS)/strlcat.o)

CLIBDEP      := $(CLIBO:.o=.d) $(CLIBDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CLIBH))

# include all dependency files
INCLUDEFILES += $(CLIBDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CLIBDIRI)/%.h
		cp $< $@

$(CLIBDS):      $(CLIBHH) $(CLIBL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CLIBHH) $(CLIBL)

all-$(MODNAME): $(CLIBO) $(CLIBDO)

clean-$(MODNAME):
		@rm -f $(CLIBO) $(CLIBDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLIBDEP) $(CLIBDS) $(CLIBDH)

distclean::     distclean-$(MODNAME)

##### extra rules ######
