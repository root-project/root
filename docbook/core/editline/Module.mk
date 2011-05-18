# Module.mk for editline module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := editline
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

EDITLINEDIR  := $(MODDIR)
EDITLINEDIRS := $(EDITLINEDIR)/src
EDITLINEDIRI := $(EDITLINEDIR)/inc

##### libEditline (part of libCore) #####
EDITLINEL    := $(MODDIRI)/LinkDef.h
EDITLINEDS   := $(call stripsrc,$(MODDIRS)/G__Editline.cxx)
EDITLINEDO   := $(EDITLINEDS:.cxx=.o)
EDITLINEDH   := $(EDITLINEDS:.cxx=.h)

EDITLINEH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
EDITLINES    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
EDITLINEO    := $(call stripsrc,$(EDITLINES:.cxx=.o))

EDITLINEDEP  := $(EDITLINEO:.o=.d) $(EDITLINEDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(EDITLINEH))

# include all dependency files
INCLUDEFILES += $(EDITLINEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(EDITLINEDIRI)/%.h
		cp $< $@

$(EDITLINEDS):  $(EDITLINEH) $(EDITLINEL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(EDITLINEH) $(EDITLINEL)

all-$(MODNAME): $(EDITLINEO) $(EDITLINEDO)

clean-$(MODNAME):
		@rm -f $(EDITLINEO) $(EDITLINEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(EDITLINEDEP) $(EDITLINEDS) $(EDITLINEDH)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(EDITLINEO):   CXXFLAGS+='-DR__CURSESHDR="$(CURSESHDR)"' $(CURSESINCDIR)
