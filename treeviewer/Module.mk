# Module.mk for treeviewer module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := treeviewer
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TREEVIEWERDIR  := $(MODDIR)
TREEVIEWERDIRS := $(TREEVIEWERDIR)/src
TREEVIEWERDIRI := $(TREEVIEWERDIR)/inc

##### libTreeViewer #####
TREEVIEWERL  := $(MODDIRI)/LinkDef.h
TREEVIEWERDS := $(MODDIRS)/G__TreeViewer.cxx
TREEVIEWERDO := $(TREEVIEWERDS:.cxx=.o)
TREEVIEWERDH := $(TREEVIEWERDS:.cxx=.h)

TREEVIEWERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TREEVIEWERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TREEVIEWERO  := $(TREEVIEWERS:.cxx=.o)

TREEVIEWERDEP := $(TREEVIEWERO:.o=.d) $(TREEVIEWERDO:.o=.d)

TREEVIEWERLIB := $(LPATH)/libTreeViewer.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(TREEVIEWERH))
ALLLIBS       += $(TREEVIEWERLIB)

# include all dependency files
INCLUDEFILES += $(TREEVIEWERDEP)

##### local rules #####
include/%.h:    $(TREEVIEWERDIRI)/%.h
		cp $< $@

$(TREEVIEWERLIB): $(TREEVIEWERO) $(TREEVIEWERDO) $(MAINLIBS) \
                  $(TREEVIEWERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTreeViewer.$(SOEXT) $@ \
		   "$(TREEVIEWERO) $(TREEVIEWERDO)" \
		   "$(TREEVIEWERLIBEXTRA)"

$(TREEVIEWERDS): $(TREEVIEWERH) $(TREEVIEWERL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TREEVIEWERH) $(TREEVIEWERL)

$(TREEVIEWERDO): $(TREEVIEWERDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-treeviewer:  $(TREEVIEWERLIB)

clean-treeviewer:
		@rm -f $(TREEVIEWERO) $(TREEVIEWERDO)

clean::         clean-treeviewer

distclean-treeviewer: clean-treeviewer
		@rm -f $(TREEVIEWERDEP) $(TREEVIEWERDS) $(TREEVIEWERDH) \
		   $(TREEVIEWERLIB)

distclean::     distclean-treeviewer
