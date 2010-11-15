# Module.mk for fitpanel module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Ilka Antcheva, 02/10/2006

MODNAME      := fitpanel
MODDIR       := $(ROOT_SRCDIR)/gui/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FITPANELDIR  := $(MODDIR)
FITPANELDIRS := $(FITPANELDIR)/src
FITPANELDIRI := $(FITPANELDIR)/inc

##### libFitPanel #####
FITPANELL    := $(MODDIRI)/LinkDef.h
FITPANELDS   := $(call stripsrc,$(MODDIRS)/G__FitPanel.cxx)
FITPANELDO   := $(FITPANELDS:.cxx=.o)
FITPANELDH   := $(FITPANELDS:.cxx=.h)

FITPANELH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FITPANELS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FITPANELO    := $(call stripsrc,$(FITPANELS:.cxx=.o))

FITPANELDEP  := $(FITPANELO:.o=.d) $(FITPANELDO:.o=.d)

FITPANELLIB  := $(LPATH)/libFitPanel.$(SOEXT)
FITPANELMAP  := $(FITPANELLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FITPANELH))
ALLLIBS      += $(FITPANELLIB)
ALLMAPS      += $(FITPANELMAP)

# include all dependency files
INCLUDEFILES += $(FITPANELDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FITPANELDIRI)/%.h
		cp $< $@

$(FITPANELLIB): $(FITPANELO) $(FITPANELDO) $(ORDER_) $(MAINLIBS) $(FITPANELLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFitPanel.$(SOEXT) $@ "$(FITPANELO) $(FITPANELDO)" \
		   "$(FITPANELLIBEXTRA)"

$(FITPANELDS):  $(FITPANELH) $(FITPANELL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FITPANELH) $(FITPANELL)

$(FITPANELMAP): $(RLIBMAP) $(MAKEFILEDEP) $(FITPANELL)
		$(RLIBMAP) -o $@ -l $(FITPANELLIB) \
		   -d $(FITPANELLIBDEPM) -c $(FITPANELL)

all-$(MODNAME): $(FITPANELLIB) $(FITPANELMAP)

clean-$(MODNAME):
		@rm -f $(FITPANELO) $(FITPANELDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FITPANELDEP) $(FITPANELDS) $(FITPANELDH) \
		   $(FITPANELLIB) $(FITPANELMAP)

distclean::     distclean-$(MODNAME)
