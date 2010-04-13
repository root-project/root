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

ifneq ($(BUILDCLING),yes)
METACL       := $(MODDIRI)/LinkDef_TCint.h
METACDS      := $(MODDIRS)/G__TCint.cxx
METACH       := $(MODDIRI)/TCint.h
else
METACL       :=
METACDS      :=
endif
METACDO      := $(METACDS:.cxx=.o)
METACDH      := $(METACDS:.cxx=.h)

METAH        := $(filter-out $(MODDIRI)/TCint.h,$(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h)))
METAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ifeq ($(BUILDCLING),yes)
METAS        := $(filter-out $(MODDIRS)/TCint.cxx,$(METAS))
endif
METAO        := $(METAS:.cxx=.o)

METADEP      := $(METAO:.o=.d) $(METADO:.o=.d) $(METACDO:.o=.d)

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

$(METACDS):     $(METACH) $(METACL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METACH) $(METACL)

all-$(MODNAME): $(METAO) $(METADO) $(METACDO)

clean-$(MODNAME):
		@rm -f $(METAO) $(METADO) $(METACDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METADEP) $(METADS) $(METADH) $(METACDS) $(METACDH)

distclean::     distclean-$(MODNAME)
