# Module.mk for proofplayer module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 16/3/2007

MODDIR       := proofplayer
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFPLAYERDIR  := $(MODDIR)
PROOFPLAYERDIRS := $(PROOFPLAYERDIR)/src
PROOFPLAYERDIRI := $(PROOFPLAYERDIR)/inc

##### libProofPlayer #####
PROOFPLAYERL  := $(MODDIRI)/LinkDef.h
PROOFPLAYERDS := $(MODDIRS)/G__ProofPlayer.cxx
PROOFPLAYERDO := $(PROOFPLAYERDS:.cxx=.o)
PROOFPLAYERDH := $(PROOFPLAYERDS:.cxx=.h)

PROOFPLAYERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PROOFPLAYERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PROOFPLAYERO  := $(PROOFPLAYERS:.cxx=.o)

PROOFPLAYERDEP := $(PROOFPLAYERO:.o=.d) $(PROOFPLAYERDO:.o=.d)

PROOFPLAYERLIB := $(LPATH)/libProofPlayer.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFPLAYERH))
ALLLIBS       += $(PROOFPLAYERLIB)

# include all dependency files
INCLUDEFILES += $(PROOFPLAYERDEP)

##### local rules #####
include/%.h:    $(PROOFPLAYERDIRI)/%.h
		cp $< $@

$(PROOFPLAYERLIB): $(PROOFPLAYERO) $(PROOFPLAYERDO) $(ORDER_) $(MAINLIBS) \
                   $(PROOFPLAYERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofPlayer.$(SOEXT) $@ \
		   "$(PROOFPLAYERO) $(PROOFPLAYERDO)" \
		   "$(PROOFPLAYERLIBEXTRA)"

$(PROOFPLAYERDS): $(PROOFPLAYERH) $(PROOFPLAYERL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PROOFPLAYERH) $(PROOFPLAYERL)

all-proofplayer: $(PROOFPLAYERLIB)

map-proofplayer: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PROOFPLAYERLIB) \
		   -d $(PROOFPLAYERLIBDEP) -c $(PROOFPLAYERL)

map::           map-proofplayer

clean-proofplayer:
		@rm -f $(PROOFPLAYERO) $(PROOFPLAYERDO)

clean::         clean-proofplayer

distclean-proofplayer: clean-proofplayer
		@rm -f $(PROOFPLAYERDEP) $(PROOFPLAYERDS) $(PROOFPLAYERDH) \
		   $(PROOFPLAYERLIB)

distclean::     distclean-proofplayer

##### extra rules ######
