# Module.mk for fitpanel module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Ilka Antcheva, 02/10/2006

MODDIR       := fitpanel
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FITPANELDIR  := $(MODDIR)
FITPANELDIRS := $(FITPANELDIR)/src
FITPANELDIRI := $(FITPANELDIR)/inc

##### libFitPanel #####
FITPANELL    := $(MODDIRI)/LinkDef.h
FITPANELDS   := $(MODDIRS)/G__FitPanel.cxx
FITPANELDO   := $(FITPANELDS:.cxx=.o)
FITPANELDH   := $(FITPANELDS:.cxx=.h)

FITPANELH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FITPANELS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FITPANELO    := $(FITPANELS:.cxx=.o)

FITPANELDEP  := $(FITPANELO:.o=.d) $(FITPANELDO:.o=.d)

FITPANELLIB  := $(LPATH)/libFitPanel.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FITPANELH))
ALLLIBS      += $(FITPANELLIB)

# include all dependency files
INCLUDEFILES += $(FITPANELDEP)

##### local rules #####
include/%.h:    $(FITPANELDIRI)/%.h
		cp $< $@

$(FITPANELLIB): $(FITPANELO) $(FITPANELDO) $(ORDER_) $(MAINLIBS) $(FITPANELLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFitPanel.$(SOEXT) $@ "$(FITPANELO) $(FITPANELDO)" \
		   "$(FITPANELLIBEXTRA)"

$(FITPANELDS):  $(FITPANELH) $(FITPANELL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FITPANELH) $(FITPANELL)

all-fitpanel:   $(FITPANELLIB)

map-fitpanel:   $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(FITPANELLIB) \
		   -d $(FITPANELLIBDEP) -c $(FITPANELL)

map::           map-fitpanel

clean-fitpanel:
		@rm -f $(FITPANELO) $(FITPANELDO)

clean::         clean-fitpanel

distclean-fitpanel: clean-fitpanel
		@rm -f $(FITPANELDEP) $(FITPANELDS) $(FITPANELDH) $(FITPANELLIB)

distclean::     distclean-fitpanel
