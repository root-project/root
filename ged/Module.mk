# Module.mk for ged module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Ilka Antcheva, 18/2/2004

MODDIR    := ged
MODDIRS   := $(MODDIR)/src
MODDIRI   := $(MODDIR)/inc

GEDDIR    := $(MODDIR)
GEDDIRS   := $(GEDDIR)/src
GEDDIRI   := $(GEDDIR)/inc

##### libGed #####
GEDL      := $(MODDIRI)/LinkDef.h
GEDDS     := $(MODDIRS)/G__Ged.cxx
GEDDO     := $(GEDDS:.cxx=.o)
GEDDH     := $(GEDDS:.cxx=.h)

GEDH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GEDS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEDO      := $(GEDS:.cxx=.o)

GEDDEP    := $(GEDO:.o=.d) $(GEDDO:.o=.d)

GEDLIB    := $(LPATH)/libGed.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEDH))
ALLLIBS     += $(GEDLIB)

# include all dependency files
INCLUDEFILES += $(GEDDEP)

##### local rules #####
include/%.h:    $(GEDDIRI)/%.h
		cp $< $@

$(GEDLIB):      $(GEDO) $(GEDDO) $(MAINLIBS) $(GEDLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGed.$(SOEXT) $@ "$(GEDO) $(GEDDO)" \
		   "$(GEDLIBEXTRA)"

$(GEDDS):       $(GEDH) $(GEDL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEDH) $(GEDL)

$(GEDDO):       $(GEDDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-ged:        $(GEDLIB)

map-ged:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GEDLIB) \
		   -d $(GEDLIBDEP) -c $(GEDL)

map::           map-ged

clean-ged:
		@rm -f $(GEDO) $(GEDDO)

clean::         clean-ged

distclean-ged: clean-ged
		@rm -f $(GEDDEP) $(GEDDS) $(GEDDH) $(GEDLIB)

distclean::     distclean-ged
