# Module.mk for sessionviewer module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 17/03/2007

MODNAME      := sessionviewer
MODDIR       := $(ROOT_SRCDIR)/gui/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SESSIONVIEWERDIR  := $(MODDIR)
SESSIONVIEWERDIRS := $(SESSIONVIEWERDIR)/src
SESSIONVIEWERDIRI := $(SESSIONVIEWERDIR)/inc

##### libSessionViewer #####
SESSIONVIEWERL  := $(MODDIRI)/LinkDef.h
SESSIONVIEWERDS := $(call stripsrc,$(MODDIRS)/G__SessionViewer.cxx)
SESSIONVIEWERDO := $(SESSIONVIEWERDS:.cxx=.o)
SESSIONVIEWERDH := $(SESSIONVIEWERDS:.cxx=.h)

SESSIONVIEWERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SESSIONVIEWERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SESSIONVIEWERO  := $(call stripsrc,$(SESSIONVIEWERS:.cxx=.o))

SESSIONVIEWERDEP := $(SESSIONVIEWERO:.o=.d) $(SESSIONVIEWERDO:.o=.d)

SESSIONVIEWERLIB := $(LPATH)/libSessionViewer.$(SOEXT)
SESSIONVIEWERMAP := $(SESSIONVIEWERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
SESSIONVIEWERH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(SESSIONVIEWERH))
ALLHDRS       += $(SESSIONVIEWERH_REL)
ALLLIBS       += $(SESSIONVIEWERLIB)
ALLMAPS       += $(SESSIONVIEWERMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(SESSIONVIEWERH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Gui_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(SESSIONVIEWERLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(SESSIONVIEWERDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SESSIONVIEWERDIRI)/%.h
		cp $< $@

$(SESSIONVIEWERLIB): $(SESSIONVIEWERO) $(SESSIONVIEWERDO) $(ORDER_) \
                     $(MAINLIBS) $(SESSIONVIEWERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSessionViewer.$(SOEXT) $@ \
		   "$(SESSIONVIEWERO) $(SESSIONVIEWERDO)" \
		   "$(SESSIONVIEWERLIBEXTRA)"

$(call pcmrule,SESSIONVIEWER)
	$(noop)

$(SESSIONVIEWERDS): $(SESSIONVIEWERH) $(SESSIONVIEWERL) $(ROOTCLINGEXE) $(call pcmdep,SESSIONVIEWER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,SESSIONVIEWER) -c $(SESSIONVIEWERH) $(SESSIONVIEWERL)

$(SESSIONVIEWERMAP): $(SESSIONVIEWERH) $(SESSIONVIEWERL) $(ROOTCLINGEXE) $(call pcmdep,SESSIONVIEWER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(SESSIONVIEWERDS) $(call dictModule,SESSIONVIEWER) -c $(SESSIONVIEWERH) $(SESSIONVIEWERL)

all-$(MODNAME): $(SESSIONVIEWERLIB)
clean-$(MODNAME):
		@rm -f $(SESSIONVIEWERO) $(SESSIONVIEWERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SESSIONVIEWERDEP) $(SESSIONVIEWERDS) \
		   $(SESSIONVIEWERDH) $(SESSIONVIEWERLIB) $(SESSIONVIEWERMAP)

distclean::     distclean-$(MODNAME)
