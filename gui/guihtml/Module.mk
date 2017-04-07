# Module.mk for guibuilder module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Valeriy Onuchin, 24/4/2007

MODNAME      := guihtml
MODDIR       := $(ROOT_SRCDIR)/gui/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GUIHTMLDIR   := $(MODDIR)
GUIHTMLDIRS  := $(GUIHTMLDIR)/src
GUIHTMLDIRI  := $(GUIHTMLDIR)/inc

##### libGuiHtml #####
GUIHTMLL     := $(MODDIRI)/LinkDef.h
GUIHTMLDS    := $(call stripsrc,$(MODDIRS)/G__GuiHtml.cxx)
GUIHTMLDO    := $(GUIHTMLDS:.cxx=.o)
GUIHTMLDH    := $(GUIHTMLDS:.cxx=.h)

GUIHTMLH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GUIHTMLS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GUIHTMLO     := $(call stripsrc,$(GUIHTMLS:.cxx=.o))

GUIHTMLDEP   := $(GUIHTMLO:.o=.d) $(GUIHTMLDO:.o=.d)

GUIHTMLLIB   := $(LPATH)/libGuiHtml.$(SOEXT)
GUIHTMLMAP   := $(GUIHTMLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
GUIHTMLH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(GUIHTMLH))
ALLHDRS     += $(GUIHTMLH_REL)
ALLLIBS     += $(GUIHTMLLIB)
ALLMAPS     += $(GUIHTMLMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(GUIHTMLH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Gui_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(GUIHTMLLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(GUIHTMLDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GUIHTMLDIRI)/%.h
		cp $< $@

$(GUIHTMLLIB):  $(GUIHTMLO) $(GUIHTMLDO) $(ORDER_) $(MAINLIBS) $(GUIHTMLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGuiHtml.$(SOEXT) $@ "$(GUIHTMLO) $(GUIHTMLDO)" \
		   "$(GUIHTMLLIBEXTRA)"

$(call pcmrule,GUIHTML)
	$(noop)

$(GUIHTMLDS):   $(GUIHTMLH) $(GUIHTMLL) $(ROOTCLINGEXE) $(call pcmdep,GUIHTML)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GUIHTML) -c $(GUIHTMLH) $(GUIHTMLL)

$(GUIHTMLMAP):  $(GUIHTMLH) $(GUIHTMLL) $(ROOTCLINGEXE) $(call pcmdep,GUIHTML)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GUIHTMLDS) $(call dictModule,GUIHTML) -c $(GUIHTMLH) $(GUIHTMLL)

all-$(MODNAME): $(GUIHTMLLIB)

clean-$(MODNAME):
		@rm -f $(GUIHTMLO) $(GUIHTMLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GUIHTMLDEP) $(GUIHTMLDS) $(GUIHTMLDH) $(GUIHTMLLIB) $(GUIHTMLMAP)

distclean::     distclean-$(MODNAME)
