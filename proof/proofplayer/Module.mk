# Module.mk for proofplayer module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 16/3/2007

MODNAME      := proofplayer
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFPLAYERDIR  := $(MODDIR)
PROOFPLAYERDIRS := $(PROOFPLAYERDIR)/src
PROOFPLAYERDIRI := $(PROOFPLAYERDIR)/inc

##### libProofPlayer #####
PROOFPLAYERL  := $(MODDIRI)/LinkDef.h
PROOFPLAYERDS := $(call stripsrc,$(MODDIRS)/G__ProofPlayer.cxx)
PROOFPLAYERDO := $(PROOFPLAYERDS:.cxx=.o)
PROOFPLAYERDH := $(PROOFPLAYERDS:.cxx=.h)

PROOFPLAYERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PROOFPLAYERH  := $(filter-out $(MODDIRI)/TProofDraw%,$(PROOFPLAYERH))
PROOFPLAYERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PROOFPLAYERS  := $(filter-out $(MODDIRS)/TProofDraw%,$(PROOFPLAYERS))
PROOFPLAYERO  := $(call stripsrc,$(PROOFPLAYERS:.cxx=.o))

PROOFPLAYERDEP := $(PROOFPLAYERO:.o=.d) $(PROOFPLAYERDO:.o=.d)

PROOFPLAYERLIB := $(LPATH)/libProofPlayer.$(SOEXT)
PROOFPLAYERMAP := $(PROOFPLAYERLIB:.$(SOEXT)=.rootmap)

##### libProofDraw #####
PROOFDRAWL   := $(MODDIRI)/LinkDefDraw.h
PROOFDRAWDS  := $(call stripsrc,$(MODDIRS)/G__ProofDraw.cxx)
PROOFDRAWDO  := $(PROOFDRAWDS:.cxx=.o)
PROOFDRAWDH  := $(PROOFDRAWDS:.cxx=.h)

PROOFDRAWH   := $(MODDIRI)/TProofDraw.h
PROOFDRAWS   := $(MODDIRS)/TProofDraw.cxx
PROOFDRAWO   := $(call stripsrc,$(PROOFDRAWS:.cxx=.o))

PROOFDRAWDEP := $(PROOFDRAWO:.o=.d) $(PROOFDRAWDO:.o=.d)

PROOFDRAWLIB := $(LPATH)/libProofDraw.$(SOEXT)
PROOFDRAWMAP := $(PROOFDRAWLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
PROOFPLAYERH_PROOFDRAWH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFPLAYERH) $(PROOFDRAWH))
ALLHDRS       += $(PROOFPLAYERH_PROOFDRAWH_REL)
ALLLIBS       += $(PROOFPLAYERLIB) $(PROOFDRAWLIB)
ALLMAPS       += $(PROOFPLAYERMAP) $(PROOFDRAWMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(PROOFPLAYERH_PROOFDRAWH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Proof_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(PROOFPLAYERLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(PROOFDRAWLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(PROOFPLAYERDEP) $(PROOFDRAWDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PROOFPLAYERDIRI)/%.h
		cp $< $@

$(PROOFPLAYERLIB): $(PROOFPLAYERO) $(PROOFPLAYERDO) $(ORDER_) $(MAINLIBS) \
                   $(PROOFPLAYERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofPlayer.$(SOEXT) $@ \
		   "$(PROOFPLAYERO) $(PROOFPLAYERDO)" \
		   "$(PROOFPLAYERLIBEXTRA)"

$(call pcmrule,PROOFPLAYER)
	$(noop)

$(PROOFPLAYERDS): $(PROOFPLAYERH) $(PROOFPLAYERL) $(ROOTCLINGEXE) $(call pcmdep,PROOFPLAYER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,PROOFPLAYER) -c $(PROOFPLAYERH) $(PROOFPLAYERL)

$(PROOFPLAYERMAP): $(PROOFPLAYERH) $(PROOFPLAYERL) $(ROOTCLINGEXE) $(call pcmdep,PROOFPLAYER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(PROOFPLAYERDS) $(call dictModule,PROOFPLAYER) -c $(PROOFPLAYERH) $(PROOFPLAYERL)

$(PROOFDRAWLIB): $(PROOFDRAWO) $(PROOFDRAWDO) $(ORDER_) $(MAINLIBS) \
                   $(PROOFDRAWLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofDraw.$(SOEXT) $@ \
		   "$(PROOFDRAWO) $(PROOFDRAWDO)" \
		   "$(PROOFDRAWLIBEXTRA)"

$(call pcmrule,PROOFDRAW)
	$(noop)

$(PROOFDRAWDS): $(PROOFDRAWH) $(PROOFDRAWL) $(ROOTCLINGEXE) $(call pcmdep,PROOFDRAW)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,PROOFDRAW) -c $(PROOFDRAWH) $(PROOFDRAWL)

$(PROOFDRAWMAP): $(PROOFDRAWH) $(PROOFDRAWL) $(ROOTCLINGEXE) $(call pcmdep,PROOFDRAW)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -f $(PROOFDRAWDS) $(call dictModule,PROOFDRAW) -c $(PROOFDRAWH) $(PROOFDRAWL)

all-$(MODNAME): $(PROOFPLAYERLIB) $(PROOFDRAWLIB)

clean-$(MODNAME):
		@rm -f $(PROOFPLAYERO) $(PROOFPLAYERDO) $(PROOFDRAWO) $(PROOFDRAWDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PROOFPLAYERDEP) $(PROOFPLAYERDS) $(PROOFPLAYERDH) \
		   $(PROOFPLAYERLIB) $(PROOFPLAYERMAP) \
		   $(PROOFDRAWDEP) $(PROOFDRAWDS) $(PROOFDRAWDH) \
		   $(PROOFDRAWLIB) $(PROOFDRAWMAP) \

distclean::     distclean-$(MODNAME)

##### extra rules ######

# Optimize dictionary with stl containers.
$(PROOFPLAYERDO): NOOPT = $(OPT)
$(PROOFDRAWDO): NOOPT = $(OPT)
