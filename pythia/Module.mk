# Module.mk for pythia module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := pythia
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PYTHIADIR    := $(MODDIR)
PYTHIADIRS   := $(PYTHIADIR)/src
PYTHIADIRI   := $(PYTHIADIR)/inc

##### libEGPythia #####
PYTHIAL      := $(MODDIRI)/LinkDef.h
PYTHIADS     := $(MODDIRS)/G__Pythia.cxx
PYTHIADO     := $(PYTHIADS:.cxx=.o)
PYTHIADH     := $(PYTHIADS:.cxx=.h)

PYTHIAH1     := $(wildcard $(MODDIRI)/T*.h)
PYTHIAH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PYTHIAS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PYTHIAO      := $(PYTHIAS:.cxx=.o)

PYTHIADEP    := $(PYTHIAO:.o=.d) $(PYTHIADO:.o=.d)

PYTHIALIB    := $(LPATH)/libEGPythia.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PYTHIAH))
ALLLIBS     += $(PYTHIALIB)

# include all dependency files
INCLUDEFILES += $(PYTHIADEP)

##### local rules #####
include/%.h:    $(PYTHIADIRI)/%.h
		cp $< $@

$(PYTHIALIB):   $(PYTHIAO) $(PYTHIADO) $(MAINLIBS) $(PYTHIALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEGPythia.$(SOEXT) $@ \
		   "$(PYTHIAO) $(PYTHIADO)" \
		   "$(PYTHIALIBEXTRA) $(FPYTHIALIBDIR) $(FPYTHIALIB)"

$(PYTHIADS):    $(PYTHIAH1) $(PYTHIAL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PYTHIAH1) $(PYTHIAL)

$(PYTHIADO):    $(PYTHIADS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-pythia:     $(PYTHIALIB)

map-pythia:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PYTHIALIB) \
		   -d $(PYTHIALIBDEP) -c $(PYTHIAL)

map::           map-pythia

clean-pythia:
		@rm -f $(PYTHIAO) $(PYTHIADO)

clean::         clean-pythia

distclean-pythia: clean-pythia
		@rm -f $(PYTHIADEP) $(PYTHIADS) $(PYTHIADH) $(PYTHIALIB)

distclean::     distclean-pythia
