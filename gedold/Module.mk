# Module.mk for gedold module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Ilka Antcheva, 18/2/2004

MODDIR       := gedold
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GEDOLDDIR    := $(MODDIR)
GEDOLDDIRS   := $(GEDOLDDIR)/src
GEDOLDDIRI   := $(GEDOLDDIR)/inc

##### libGedOld #####
GEDOLDL      := $(MODDIRI)/LinkDef.h
GEDOLDDS     := $(MODDIRS)/G__GedOld.cxx
GEDOLDDO     := $(GEDOLDDS:.cxx=.o)
GEDOLDDH     := $(GEDOLDDS:.cxx=.h)

GEDOLDH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GEDOLDS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEDOLDO      := $(GEDOLDS:.cxx=.o)

GEDOLDDEP    := $(GEDOLDO:.o=.d) $(GEDOLDDO:.o=.d)

GEDOLDLIB    := $(LPATH)/libGedOld.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEDOLDH))
ALLLIBS     += $(GEDOLDLIB)

# include all dependency files
INCLUDEFILES += $(GEDOLDDEP)

##### local rules #####
include/%.h:    $(GEDOLDDIRI)/%.h
		cp $< $@

$(GEDOLDLIB):   $(GEDOLDO) $(GEDOLDDO) $(MAINLIBS) $(GEDOLDLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGedOld.$(SOEXT) $@ "$(GEDOLDO) $(GEDOLDDO)" \
		   "$(GEDOLDLIBEXTRA)"

$(GEDOLDDS):    $(GEDOLDH) $(GEDOLDL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEDOLDH) $(GEDOLDL)

$(GEDOLDDO):    $(GEDOLDDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-gedold:     $(GEDOLDLIB)

map-gedold:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GEDOLDLIB) \
		   -d $(GEDOLDLIBDEP) -c $(GEDOLDL)

map::           map-gedold

clean-gedold:
		@rm -f $(GEDOLDO) $(GEDOLDDO)

clean::         clean-gedold

distclean-gedold: clean-gedold
		@rm -f $(GEDOLDDEP) $(GEDOLDDS) $(GEDOLDDH) $(GEDOLDLIB)

distclean::     distclean-gedold
