# Module.mk for venus module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := venus
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

VENUSDIR     := $(MODDIR)
VENUSDIRS    := $(VENUSDIR)/src
VENUSDIRI    := $(VENUSDIR)/inc

##### libEGVenus #####
VENUSL       := $(MODDIRI)/LinkDef.h
VENUSDS      := $(MODDIRS)/G__Venus.cxx
VENUSDO      := $(VENUSDS:.cxx=.o)
VENUSDH      := $(VENUSDS:.cxx=.h)

VENUSH1      := $(wildcard $(MODDIRI)/T*.h)
VENUSH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
VENUSS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
VENUSO       := $(VENUSS:.cxx=.o)

VENUSDEP     := $(VENUSO:.o=.d) $(VENUSDO:.o=.d)

VENUSLIB     := $(LPATH)/libEGVenus.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(VENUSH))
ALLLIBS     += $(VENUSLIB)

# include all dependency files
INCLUDEFILES += $(VENUSDEP)

##### local rules #####
include/%.h:    $(VENUSDIRI)/%.h
		cp $< $@

$(VENUSLIB):    $(VENUSO) $(VENUSDO) $(MAINLIBS) $(VENUSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEGVenus.$(SOEXT) $@ \
		   "$(VENUSO) $(VENUSDO)" \
		   "$(VENUSLIBEXTRA) $(FVENUSLIBDIR) $(FVENUSLIB)"

$(VENUSDS):     $(VENUSH1) $(VENUSL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(VENUSH1) $(VENUSL)

$(VENUSDO):     $(VENUSDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-venus:      $(VENUSLIB)

map-venus:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(VENUSLIB) \
		   -d $(VENUSLIBDEP) -c $(VENUSL)

map::           map-venus

clean-venus:
		@rm -f $(VENUSO) $(VENUSDO)

clean::         clean-venus

distclean-venus: clean-venus
		@rm -f $(VENUSDEP) $(VENUSDS) $(VENUSDH) $(VENUSLIB)

distclean::     distclean-venus
