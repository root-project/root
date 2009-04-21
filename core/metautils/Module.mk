# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Philippe Canal 9/1/2004

MODNAME        := metautils
MODDIR         := core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

METAUTILSDIR   := $(MODDIR)
METAUTILSDIRS  := $(METAUTILSDIR)/src
METAUTILSDIRI  := $(METAUTILSDIR)/inc

##### $(METAUTILSO) #####
METAUTILSH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
METAUTILSS     := $(filter-out %7.cxx,$(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))
METAUTILSO     := $(METAUTILSS:.cxx=.o)

ifneq ($(BUILDBOTHCINT),)
METAUTILS7S     := $(patsubst %.cxx,%7.cxx,$(METAUTILSS))
METAUTILS7O     := $(METAUTILS7S:.cxx=.o)
endif

METAUTILSL     := $(MODDIRI)/LinkDef.h
METAUTILSDS    := $(MODDIRS)/G__MetaUtils.cxx
METAUTILSDO    := $(METAUTILSDS:.cxx=.o)
METAUTILSDH    := $(METAUTILSDS:.cxx=.h)

METAUTILSDEP   := $(METAUTILSO:.o=.d) $(METAUTILSDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAUTILSH))

# include all dependency files
INCLUDEFILES += $(METAUTILSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METAUTILSDIRI)/%.h
		cp $< $@

$(METAUTILSDS): $(METAUTILSH) $(METAUTILSL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METAUTILSH) $(METAUTILSL)

all-$(MODNAME): $(METAUTILSO) $(METAUTILSDO)

clean-$(MODNAME):
		@rm -f $(METAUTILSO) $(METAUTILSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METAUTILSDEP) $(METAUTILSDS) $(METAUTILSDH)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(METAUTILSO):  PCHCXXFLAGS =
ifneq ($(BUILDBOTHCINT),)
$(METAUTILS7O): CXXFLAGS += -DR__BUILDING_CINT7
endif
