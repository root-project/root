# Module.mk for tree module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := tree
MODDIR       := tree/$(MODNAME)
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

# ManualBase4 only needs to be regenerated (and then changed manually) when
# the dictionary interface changes
TREEL2       := $(MODDIRI)/LinkDef2.h
TREEDS2      := $(MODDIRS)/ManualTree2.cxx
TREEDO2      := $(TREEDS2:.cxx=.o)
TREEDH2      := TTree.h

TREEH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TREES        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TREEO        := $(TREES:.cxx=.o)

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
$(TREEDO2): CXXFLAGS += -Iinclude/cint

.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TREEDIRI)/%.h
		cp $< $@

$(TREELIB):     $(TREEO) $(TREEDO) $(ORDER_) $(MAINLIBS) $(TREELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTree.$(SOEXT) $@ "$(TREEO) $(TREEDO)" \
		   "$(TREELIBEXTRA)"

$(TREEDS):      $(TREEH) $(TREEL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TREEH) $(TREEL)

# pre-requisites intentionally not specified... should be called only
# on demand after deleting the file
$(TREEDS2):
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TREEDH2) $(TREEL2)

$(TREEMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(TREEL)
		$(RLIBMAP) -o $(TREEMAP) -l $(TREELIB) \
		   -d $(TREELIBDEPM) -c $(TREEL)

all-$(MODNAME): $(TREELIB) $(TREEMAP)

clean-$(MODNAME):
		@rm -f $(TREEO) $(TREEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TREEDEP) $(TREEDS) $(TREEDH) $(TREELIB) $(TREEMAP)

distclean::     distclean-$(MODNAME)
