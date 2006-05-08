# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Module.mk for tmva module

MODDIR       := tmva
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TMVADIR      := $(MODDIR)
TMVADIRS     := $(TMVADIR)/src
TMVADIRI     := $(TMVADIR)/inc

##### libTMVA #####
TMVAL        := $(MODDIRI)/LinkDef.h
TMVADS       := $(MODDIRS)/G__TMVA.cxx
TMVADO       := $(TMVADS:.cxx=.o)
TMVADH       := $(TMVADS:.cxx=.h)

TMVAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TMVAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TMVAO        := $(TMVAS:.cxx=.o)

TMVADEP      := $(TMVAO:.o=.d) $(TMVADO:.o=.d)

TMVALIB      := $(LPATH)/libTMVA.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TMVAH))
ALLLIBS      += $(TMVALIB)

# include all dependency files
INCLUDEFILES += $(TMVADEP)

##### local rules #####
include/%.h:    $(TMVADIRI)/%.h
		cp $< $@

$(TMVALIB):     $(TMVAO) $(TMVADO) $(ORDER_) $(MAINLIBS) $(TMVALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTMVA.$(SOEXT) $@ "$(TMVAO) $(TMVADO)" \
		   "$(TMVALIBEXTRA)"

$(TMVADS):      $(TMVAH) $(TMVAL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TMVAH) $(TMVAL)

all-tmva:       $(TMVALIB)

map-tmva:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(TMVALIB) \
		   -d $(TMVALIBDEP) -c $(TMVAL)

map::           map-tmva

clean-tmva:
		@rm -f $(TMVAO) $(TMVADO)

clean::         clean-tmva

distclean-tmva: clean-tmva
		@rm -f $(TMVADEP) $(TMVADS) $(TMVADH) $(TMVALIB)

distclean::     distclean-tmva
