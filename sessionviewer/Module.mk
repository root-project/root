# Module.mk for sessionviewer module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 17/03/2007

MODDIR       := sessionviewer
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SESSIONVIEWERDIR  := $(MODDIR)
SESSIONVIEWERDIRS := $(SESSIONVIEWERDIR)/src
SESSIONVIEWERDIRI := $(SESSIONVIEWERDIR)/inc

##### libSessionViewer #####
SESSIONVIEWERL  := $(MODDIRI)/LinkDef.h
SESSIONVIEWERDS := $(MODDIRS)/G__SessionViewer.cxx
SESSIONVIEWERDO := $(SESSIONVIEWERDS:.cxx=.o)
SESSIONVIEWERDH := $(SESSIONVIEWERDS:.cxx=.h)

SESSIONVIEWERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SESSIONVIEWERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SESSIONVIEWERO  := $(SESSIONVIEWERS:.cxx=.o)

SESSIONVIEWERDEP := $(SESSIONVIEWERO:.o=.d) $(SESSIONVIEWERDO:.o=.d)

SESSIONVIEWERLIB := $(LPATH)/libSessionViewer.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SESSIONVIEWERH))
ALLLIBS       += $(SESSIONVIEWERLIB)

# include all dependency files
INCLUDEFILES += $(SESSIONVIEWERDEP)

##### local rules #####
include/%.h:    $(SESSIONVIEWERDIRI)/%.h
		cp $< $@

$(SESSIONVIEWERLIB): $(SESSIONVIEWERO) $(SESSIONVIEWERDO) $(ORDER_) \
                     $(MAINLIBS) $(SESSIONVIEWERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSessionViewer.$(SOEXT) $@ \
		   "$(SESSIONVIEWERO) $(SESSIONVIEWERDO)" \
		   "$(SESSIONVIEWERLIBEXTRA)"

$(SESSIONVIEWERDS): $(SESSIONVIEWERH) $(SESSIONVIEWERL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(SESSIONVIEWERH) $(SESSIONVIEWERL)

all-sessionviewer: $(SESSIONVIEWERLIB)

map-sessionviewer: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(SESSIONVIEWERLIB) \
		   -d $(SESSIONVIEWERLIBDEP) -c $(SESSIONVIEWERL)

map::           map-sessionviewer

clean-sessionviewer:
		@rm -f $(SESSIONVIEWERO) $(SESSIONVIEWERDO)

clean::         clean-sessionviewer

distclean-sessionviewer: clean-sessionviewer
		@rm -f $(SESSIONVIEWERDEP) $(SESSIONVIEWERDS) \
		   $(SESSIONVIEWERDH) $(SESSIONVIEWERLIB)

distclean::     distclean-sessionviewer
