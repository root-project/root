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

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FUMILIH))
ALLLIBS     += $(FUMILILIB)

# include all dependency files
INCLUDEFILES += $(FUMILIDEP)

##### local rules #####
include/%.h:    $(FUMILIDIRI)/%.h
		cp $< $@

$(FUMILILIB):   $(FUMILIO) $(FUMILIDO) $(MAINLIBS) $(FUMILILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFumili.$(SOEXT) $@ "$(FUMILIO) $(FUMILIDO)" \
		   "$(FUMILILIBEXTRA)"

$(FUMILIDS):    $(FUMILIH) $(FUMILIL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FUMILIH) $(FUMILIL)

$(FUMILIDO):    $(FUMILIDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-fumili:     $(FUMILILIB)

map-fumili:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(FUMILILIB) \
		   -d $(FUMILILIBDEP) -c $(FUMILIL)

map::           map-fumili

clean-fumili:
		@rm -f $(FUMILIO) $(FUMILIDO)

clean::         clean-fumili

distclean-fumili: clean-fumili
		@rm -f $(FUMILIDEP) $(FUMILIDS) $(FUMILIDH) $(FUMILILIB)

distclean::     distclean-fumili
