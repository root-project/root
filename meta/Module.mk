# Module.mk for meta module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := meta
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
METAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
METAO        := $(METAS:.cxx=.o)

METADEP      := $(METAO:.o=.d) $(METADO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAH))

# include all dependency files
INCLUDEFILES += $(METADEP)

##### local rules #####
include/%.h:    $(METADIRI)/%.h
		cp $< $@

$(METADS):      $(METAH) $(METAL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METAH) $(METAL)

$(METADO):      $(METADS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $(METADO) -c $(METADS)

all-meta:       $(METAO) $(METADO)

clean-meta:
		@rm -f $(METAO) $(METADO)

clean::         clean-meta

distclean-meta: clean-meta
		@rm -f $(METADEP) $(METADS) $(METADH)

distclean::     distclean-meta

##### extra rules ######
ifeq ($(VC_MAJOR),13)
$(METADIRS)/TStreamerInfo.o: $(METADIRS)/TStreamerInfo.cxx 
	$(CXX) -O $(CXXFLAGS) -o $@ -c $< 
endif
