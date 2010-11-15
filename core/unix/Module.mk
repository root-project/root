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
UNIXDS       := $(call stripsrc,$(MODDIRS)/G__Unix.cxx)
UNIXDO       := $(UNIXDS:.cxx=.o)
UNIXDH       := $(UNIXDS:.cxx=.h)

UNIXH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
UNIXS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
UNIXO        := $(call stripsrc,$(UNIXS:.cxx=.o))

UNIXDEP      := $(UNIXO:.o=.d) $(UNIXDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(UNIXH))

# include all dependency files
INCLUDEFILES += $(UNIXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(UNIXDIRI)/%.h
		cp $< $@

$(UNIXDS):      $(UNIXH) $(UNIXL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(UNIXH) $(UNIXL)

all-$(MODNAME): $(UNIXO) $(UNIXDO)

clean-$(MODNAME):
		@rm -f $(UNIXO) $(UNIXDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(UNIXDEP) $(UNIXDS) $(UNIXDH)

distclean::     distclean-$(MODNAME)
