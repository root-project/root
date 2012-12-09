# Module.mk for tree module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := tree
MODDIR       := $(ROOT_SRCDIR)/tree/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TREEDIR      := $(MODDIR)
TREEDIRS     := $(TREEDIR)/src
TREEDIRI     := $(TREEDIR)/inc

##### libTree #####
TREEL        := $(MODDIRI)/LinkDef.h
TREEDS       := $(call stripsrc,$(MODDIRS)/G__Tree.cxx)
TREEDO       := $(TREEDS:.cxx=.o)
TREEDH       := $(TREEDS:.cxx=.h)

TREEH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TREES        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TREEO        := $(call stripsrc,$(TREES:.cxx=.o))

TREEDEP      := $(TREEO:.o=.d) $(TREEDO:.o=.d)

TREELIB      := $(LPATH)/libTree.$(SOEXT)
TREEMAP      := $(TREELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TREEH))
ALLLIBS     += $(TREELIB)
ALLMAPS     += $(TREEMAP)

# include all dependency files
INCLUDEFILES += $(TREEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TREEDIRI)/%.h
		cp $< $@

$(TREELIB):     $(TREEO) $(TREEDO) $(ORDER_) $(MAINLIBS) $(TREELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTree.$(SOEXT) $@ "$(TREEO) $(TREEDO)" \
		   "$(TREELIBEXTRA)"

$(TREEDS):      $(TREEH) $(TREEL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TREEH) $(TREEL)

$(TREEMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(TREEL)
		$(RLIBMAP) -o $@ -l $(TREELIB) \
		   -d $(TREELIBDEPM) -c $(TREEL)

all-$(MODNAME): $(TREELIB) $(TREEMAP)

clean-$(MODNAME):
		@rm -f $(TREEO) $(TREEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TREEDEP) $(TREEDS) $(TREEDH) $(TREELIB) $(TREEMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
