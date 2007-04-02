# Module.mk for fumili module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 07/05/2003

MODDIR       := fumili
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FUMILIDIR    := $(MODDIR)
FUMILIDIRS   := $(FUMILIDIR)/src
FUMILIDIRI   := $(FUMILIDIR)/inc

##### libFumili #####
FUMILIL      := $(MODDIRI)/LinkDef.h
FUMILIDS     := $(MODDIRS)/G__Fumili.cxx
FUMILIDO     := $(FUMILIDS:.cxx=.o)
FUMILIDH     := $(FUMILIDS:.cxx=.h)

FUMILIH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FUMILIS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FUMILIO      := $(FUMILIS:.cxx=.o)

FUMILIDEP    := $(FUMILIO:.o=.d) $(FUMILIDO:.o=.d)

FUMILILIB    := $(LPATH)/libFumili.$(SOEXT)
FUMILIMAP    := $(FUMILILIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FUMILIH))
ALLLIBS     += $(FUMILILIB)
ALLMAPS     += $(FUMILIMAP)

# include all dependency files
INCLUDEFILES += $(FUMILIDEP)

##### local rules #####
include/%.h:    $(FUMILIDIRI)/%.h
		cp $< $@

$(FUMILILIB):   $(FUMILIO) $(FUMILIDO) $(ORDER_) $(MAINLIBS) $(FUMILILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFumili.$(SOEXT) $@ "$(FUMILIO) $(FUMILIDO)" \
		   "$(FUMILILIBEXTRA)"

$(FUMILIDS):    $(FUMILIH) $(FUMILIL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FUMILIH) $(FUMILIL)

$(FUMILIMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(FUMILIL)
		$(RLIBMAP) -o $(FUMILIMAP) -l $(FUMILILIB) \
		   -d $(FUMILILIBDEPM) -c $(FUMILIL)

all-fumili:     $(FUMILILIB) $(FUMILIMAP)

clean-fumili:
		@rm -f $(FUMILIO) $(FUMILIDO)

clean::         clean-fumili

distclean-fumili: clean-fumili
		@rm -f $(FUMILIDEP) $(FUMILIDS) $(FUMILIDH) $(FUMILILIB) $(FUMILIMAP)

distclean::     distclean-fumili
