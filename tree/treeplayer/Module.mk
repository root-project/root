# Module.mk for treeplayer module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := treeplayer
MODDIR       := tree/$(MODNAME)
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
TREEPLAYERMAP := $(TREEPLAYERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TREEPLAYERH))
ALLLIBS       += $(TREEPLAYERLIB)
ALLMAPS       += $(TREEPLAYERMAP)

# include all dependency files
INCLUDEFILES += $(TREEPLAYERDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TREEPLAYERDIRI)/%.h
		cp $< $@

$(TREEPLAYERLIB): $(TREEPLAYERO) $(TREEPLAYERDO) $(ORDER_) $(MAINLIBS) \
                  $(TREEPLAYERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTreePlayer.$(SOEXT) $@ \
		   "$(TREEPLAYERO) $(TREEPLAYERDO)" \
		   "$(TREEPLAYERLIBEXTRA)"

$(TREEPLAYERDS): $(TREEPLAYERH) $(TREEPLAYERL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TREEPLAYERH) $(TREEPLAYERL)

$(TREEPLAYERMAP): $(RLIBMAP) $(MAKEFILEDEP) $(TREEPLAYERL)
		$(RLIBMAP) -o $(TREEPLAYERMAP) -l $(TREEPLAYERLIB) \
		   -d $(TREEPLAYERLIBDEPM) -c $(TREEPLAYERL)

all-$(MODNAME): $(TREEPLAYERLIB) $(TREEPLAYERMAP)

clean-$(MODNAME):
		@rm -f $(TREEPLAYERO) $(TREEPLAYERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TREEPLAYERDEP) $(TREEPLAYERDS) $(TREEPLAYERDH) \
		   $(TREEPLAYERLIB) $(TREEPLAYERMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(PLATFORM),macosx)
ifeq ($(GCC_VERS_FULL),gcc-4.0.1)
ifneq ($(filter -O%,$(OPT)),)
   $(TREEPLAYERDIRS)/TTreeFormula.o: OPT = $(NOOPT)
endif
endif
ifeq ($(ICC_MAJOR),10)
$(TREEPLAYERDIRS)/TTreeFormula.o: OPT = $(NOOPT)
endif
endif
