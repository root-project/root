# Module.mk for tree module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := tree
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TREEDIR      := $(MODDIR)
TREEDIRS     := $(TREEDIR)/src
TREEDIRI     := $(TREEDIR)/inc

##### libTree #####
TREEL        := $(MODDIRI)/LinkDef.h
TREEDS       := $(MODDIRS)/G__Tree.cxx
TREEDO       := $(TREEDS:.cxx=.o)
TREEDH       := $(TREEDS:.cxx=.h)

TREEH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TREES        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TREEO        := $(TREES:.cxx=.o)

TREEDEP      := $(TREEO:.o=.d) $(TREEDO:.o=.d)

TREELIB      := $(LPATH)/libTree.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TREEH))
ALLLIBS     += $(TREELIB)

# include all dependency files
INCLUDEFILES += $(TREEDEP)

##### local rules #####
include/%.h:    $(TREEDIRI)/%.h
		cp $< $@

$(TREELIB):     $(TREEO) $(TREEDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTree.$(SOEXT) $@ "$(TREEO) $(TREEDO)" \
		   "$(TREELIBEXTRA)"

$(TREEDS):      $(TREEH) $(TREEL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TREEH) $(TREEL)

$(TREEDO):      $(TREEDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-tree:       $(TREELIB)

clean-tree:
		@rm -f $(TREEO) $(TREEDO)

clean::         clean-tree

distclean-tree: clean-tree
		@rm -f $(TREEDEP) $(TREEDS) $(TREEDH) $(TREELIB)

distclean::     distclean-tree
