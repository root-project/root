# Module.mk for treeplayer module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := treeplayer
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TREEPLAYERDIR  := $(MODDIR)
TREEPLAYERDIRS := $(TREEPLAYERDIR)/src
TREEPLAYERDIRI := $(TREEPLAYERDIR)/inc

##### libTreePlayer #####
TREEPLAYERL  := $(MODDIRI)/LinkDef.h
TREEPLAYERDS := $(MODDIRS)/G__TreePlayer.cxx
TREEPLAYERDO := $(TREEPLAYERDS:.cxx=.o)
TREEPLAYERDH := $(TREEPLAYERDS:.cxx=.h)

TREEPLAYERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TREEPLAYERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TREEPLAYERO  := $(TREEPLAYERS:.cxx=.o)

TREEPLAYERDEP := $(TREEPLAYERO:.o=.d) $(TREEPLAYERDO:.o=.d)

TREEPLAYERLIB := $(LPATH)/libTreePlayer.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TREEPLAYERH))
ALLLIBS       += $(TREEPLAYERLIB)

# include all dependency files
INCLUDEFILES += $(TREEPLAYERDEP)

##### local rules #####
include/%.h:    $(TREEPLAYERDIRI)/%.h
		cp $< $@

$(TREEPLAYERLIB): $(TREEPLAYERO) $(TREEPLAYERDO) $(MAINLIBS) \
                  $(TREEPLAYERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTreePlayer.$(SOEXT) $@ \
		   "$(TREEPLAYERO) $(TREEPLAYERDO)" \
		   "$(TREEPLAYERLIBEXTRA)"

$(TREEPLAYERDS): $(TREEPLAYERH) $(TREEPLAYERL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TREEPLAYERH) $(TREEPLAYERL)

$(TREEPLAYERDO): $(TREEPLAYERDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-treeplayer: $(TREEPLAYERLIB)

map-treeplayer: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(TREEPLAYERLIB) \
		   -d $(TREEPLAYERLIBDEP) -c $(TREEPLAYERL)

map::           map-treeplayer

clean-treeplayer:
		@rm -f $(TREEPLAYERO) $(TREEPLAYERDO)

clean::         clean-treeplayer

distclean-treeplayer: clean-treeplayer
		@rm -f $(TREEPLAYERDEP) $(TREEPLAYERDS) $(TREEPLAYERDH) \
		   $(TREEPLAYERLIB)

distclean::     distclean-treeplayer
