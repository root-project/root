# Module.mk for macosx module
# Copyright (c) 2011 Rene Brun and Fons Rademakers
#
# Author: Timur Pocheptsov, 5/12/2011

MODNAME      := macosx
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MACOSXDIR    := $(MODDIR)
MACOSXDIRS   := $(MACOSXDIR)/src
MACOSXDIRI   := $(MACOSXDIR)/inc

##### libMacOSX  (part of libCore) #####
MACOSXL	     := $(MODDIRI)/LinkDef.h
MACOSXDS     := $(call stripsrc,$(MODDIRS)/G__Macosx.cxx)
MACOSXDO     := $(MACOSXDS:.cxx=.o)
MACOSXDH     := $(MACOSXDS:.cxx=.h)

MACOSXH1     := $(wildcard $(MODDIRI)/T*.h)
MACOSXH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MACOSXS      := $(wildcard $(MODDIRS)/*.mm)
MACOSXO      := $(call stripsrc,$(MACOSXS:.mm=.o))

MACOSXDEP    := $(MACOSXO:.o=.d) $(MACOSXDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MACOSXH))

# include all dependency files
INCLUDEFILES += $(MACOSXDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MACOSXDIRI)/%.h
		cp $< $@

$(MACOSXDS):    $(MACOSXH1) $(MACOSXL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MACOSXH1) $(MACOSXL)

all-$(MODNAME): $(MACOSXO) $(MACOSXDO)

clean-$(MODNAME):
		@rm -f $(MACOSXO) $(MACOSXDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MACOSXDEP) $(MACOSXDS) $(MACOSXDH)

distclean::     distclean-$(MODNAME)
