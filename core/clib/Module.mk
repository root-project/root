# Module.mk for clib module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := clib
MODDIR       := core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CLIBDIR      := $(MODDIR)
CLIBDIRS     := $(CLIBDIR)/src
CLIBDIRI     := $(CLIBDIR)/inc

##### libClib (part of libCore) #####
CLIBL        := $(MODDIRI)/LinkDef.h
CLIBDS       := $(MODDIRS)/G__Clib.cxx
CLIBDO       := $(CLIBDS:.cxx=.o)
CLIBDH       := $(CLIBDS:.cxx=.h)

CLIBH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CLIBS1       := $(wildcard $(MODDIRS)/*.c)
ifeq ($(BUILDEDITLINE),yes)
CLIBS1       := $(filter-out $(MODDIRS)/Getline.c,$(CLIBS1))
endif
CLIBS2       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

CLIBO        := $(CLIBS1:.c=.o) $(CLIBS2:.cxx=.o)
SNPRINTFO    := $(CLIBDIRS)/snprintf.o

CLIBDEP      := $(CLIBO:.o=.d) $(CLIBDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CLIBH))

# include all dependency files
INCLUDEFILES += $(CLIBDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CLIBDIRI)/%.h
		cp $< $@

$(CLIBDS):      $(CLIBDIRI)/Getline.h $(CLIBL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CLIBDIRI)/Getline.h $(CLIBL)

all-$(MODNAME): $(CLIBO) $(CLIBDO)

clean-$(MODNAME):
		@rm -f $(CLIBO) $(CLIBDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLIBDEP) $(CLIBDS) $(CLIBDH)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(CLIBO):       PCHCXXFLAGS =
