# Module.mk for star module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := star
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

STARDIR      := $(MODDIR)
STARDIRS     := $(STARDIR)/src
STARDIRI     := $(STARDIR)/inc

##### libStar #####
STARL        := $(MODDIRI)/LinkDef.h
STARDS       := $(MODDIRS)/G__Star.cxx
STARDO       := $(STARDS:.cxx=.o)
STARDH       := $(STARDS:.cxx=.h)

STARH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
STARS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
STARO        := $(STARS:.cxx=.o)

STARDEP      := $(STARO:.o=.d) $(STARDO:.o=.d)

STARLIB      := $(LPATH)/libStar.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(STARH))
ALLLIBS     += $(STARLIB)

# include all dependency files
INCLUDEFILES += $(STARDEP)

##### local rules #####
include/%.h:    $(STARDIRI)/%.h
		cp $< $@

$(STARLIB):     $(STARO) $(STARDO) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libStar.$(SOEXT) $@ "$(STARO) $(STARDO)" \
		   "$(STARLIBEXTRA)"

$(STARDS):      $(STARH) $(STARL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(STARH) $(STARL)

$(STARDO):      $(STARDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-star:       $(STARLIB)

clean-star:
		@rm -f $(STARO) $(STARDO)

clean::         clean-star

distclean-star: clean-star
		@rm -f $(STARDEP) $(STARDS) $(STARDH) $(STARLIB)

distclean::     distclean-star
