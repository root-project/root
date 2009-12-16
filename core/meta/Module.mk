# Module.mk for meta module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := meta
MODDIR       := core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

METADIR      := $(MODDIR)
METADIRS     := $(METADIR)/src
METADIRI     := $(METADIR)/inc

##### libMeta (part of libCore) #####
METAL        := $(MODDIRI)/LinkDef.h
METADS       := $(MODDIRS)/G__Meta.cxx
METADO       := $(METADS:.cxx=.o)
METADH       := $(METADS:.cxx=.h)

METAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
METAH        := $(METAH)
METAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
METAS        := $(METAS)
METAO        := $(METAS:.cxx=.o)

METADEP      := $(METAO:.o=.d) $(METADO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAH))

# include all dependency files
INCLUDEFILES += $(METADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METADIRI)/%.h
		cp $< $@

$(METADS):      $(METAH) $(METAL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METAH) $(METAL)

all-$(MODNAME): $(METAO) $(METADO)

clean-$(MODNAME):
		@rm -f $(METAO) $(METADO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METADEP) $(METADS) $(METADH)

distclean::     distclean-$(MODNAME)
