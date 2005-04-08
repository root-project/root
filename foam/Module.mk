# Module.mk for foam module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := foam
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FOAMDIR      := $(MODDIR)
FOAMDIRS     := $(FOAMDIR)/src
FOAMDIRI     := $(FOAMDIR)/inc

##### libFoam.so #####
FOAML      := $(MODDIRI)/LinkDef.h
FOAMDS     := $(MODDIRS)/G__Foam.cxx
FOAMDO     := $(FOAMDS:.cxx=.o)
FOAMDH     := $(FOAMDS:.cxx=.h)

FOAMH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FOAMS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FOAMO      := $(FOAMS:.cxx=.o)

FOAMDEP    := $(FOAMO:.o=.d) $(FOAMDO:.o=.d)

FOAMLIB    := $(LPATH)/libFoam.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FOAMH))
ALLLIBS     += $(FOAMLIB)

# include all dependency files
INCLUDEFILES += $(FOAMDEP)

##### local rules #####
include/%.h:    $(FOAMDIRI)/%.h
		cp $< $@

$(FOAMLIB):     $(FOAMO) $(FOAMDO) $(MAINLIBS) $(FOAMLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFoam.$(SOEXT) $@ "$(FOAMO) $(FOAMDO)" \
		   "$(FOAMLIBEXTRA)"

$(FOAMDS):      $(FOAMH) $(FOAML) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FOAMH) $(FOAML)

$(FOAMDO):      $(FOAMDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-foam:       $(FOAMLIB)

map-foam:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(FOAMLIB) \
		   -d $(FOAMLIBDEP) -c $(FOAML)

map::           map-foam

clean-foam:
		@rm -f $(FOAMO) $(FOAMDO)

clean::         clean-foam

distclean-foam: clean-foam
		@rm -f $(FOAMDEP) $(FOAMDS) $(FOAMDH) $(FOAMLIB)

distclean::     distclean-foam
