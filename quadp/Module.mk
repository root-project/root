# Module.mk for quadp module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, Eddy Offermann, 21/05/2003

MODDIR       := quadp
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

QUADPDIR     := $(MODDIR)
QUADPDIRS    := $(QUADPDIR)/src
QUADPDIRI    := $(QUADPDIR)/inc

##### libQuadp #####
QUADPL       := $(MODDIRI)/LinkDef.h
QUADPDS      := $(MODDIRS)/G__Quadp.cxx
QUADPDO      := $(QUADPDS:.cxx=.o)
QUADPDH      := $(QUADPDS:.cxx=.h)

QUADPH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
QUADPS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
QUADPO       := $(QUADPS:.cxx=.o)

QUADPDEP     := $(QUADPO:.o=.d) $(QUADPDO:.o=.d)

QUADPLIB     := $(LPATH)/libQuadp.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(QUADPH))
ALLLIBS     += $(QUADPLIB)

# include all dependency files
INCLUDEFILES += $(QUADPDEP)

##### local rules #####
include/%.h:    $(QUADPDIRI)/%.h
		cp $< $@

$(QUADPLIB):    $(QUADPO) $(QUADPDO) $(MAINLIBS) $(QUADPLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libQuadp.$(SOEXT) $@ "$(QUADPO) $(QUADPDO)" \
		   "$(QUADPLIBEXTRA)"

$(QUADPDS):     $(QUADPH) $(QUADPL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(QUADPH) $(QUADPL)

$(QUADPDO):     $(QUADPDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-quadp:      $(QUADPLIB)

map-quadp:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(QUADPLIB) \
		   -d $(QUADPLIBDEP) -c $(QUADPL)

map::           map-quadp

clean-quadp:
		@rm -f $(QUADPO) $(QUADPDO)

clean::         clean-quadp

distclean-quadp: clean-quadp
		@rm -f $(QUADPDEP) $(QUADPDS) $(QUADPDH) $(QUADPLIB)

distclean::     distclean-quadp
