# Module.mk for graf module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := graf
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GRAFDIR      := $(MODDIR)
GRAFDIRS     := $(GRAFDIR)/src
GRAFDIRI     := $(GRAFDIR)/inc

##### libGraf #####
GRAFL1       := $(MODDIRI)/LinkDef1.h
GRAFL2       := $(MODDIRI)/LinkDef2.h
GRAFDS1      := $(MODDIRS)/G__Graf1.cxx
GRAFDS2      := $(MODDIRS)/G__Graf2.cxx
GRAFDO1      := $(GRAFDS1:.cxx=.o)
GRAFDO2      := $(GRAFDS2:.cxx=.o)
GRAFDS       := $(GRAFDS1) $(GRAFDS2)
GRAFDO       := $(GRAFDO1) $(GRAFDO2)
GRAFDH       := $(GRAFDS:.cxx=.h)

GRAFH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GRAFS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GRAFO        := $(GRAFS:.cxx=.o)

GRAFDEP      := $(GRAFO:.o=.d)

GRAFLIB      := $(LPATH)/libGraf.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GRAFH))
ALLLIBS     += $(GRAFLIB)

# include all dependency files
INCLUDEFILES += $(GRAFDEP)

##### local rules #####
include/%.h:    $(GRAFDIRI)/%.h
		cp $< $@

$(GRAFLIB):     $(GRAFO) $(GRAFDO) $(MAINLIBS) $(HISTLIB)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGraf.$(SOEXT) $@ "$(GRAFO) $(GRAFDO)" \
		   "$(GRAFLIBEXTRA)"

$(GRAFDS1):     $(GRAFH) $(GRAFL1) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		@$(ROOTCINTTMP) -f $@ -c $(GRAFH) $(GRAFL1)
$(GRAFDS2):     $(GRAFH) $(GRAFL2) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		@$(ROOTCINTTMP) -f $@ -c $(GRAFH) $(GRAFL2)

$(GRAFDO1):     $(GRAFDS1)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<
$(GRAFDO2):     $(GRAFDS2)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-graf:       $(GRAFLIB)

clean-graf:
		@rm -f $(GRAFO) $(GRAFDO)

clean::         clean-graf

distclean-graf: clean-graf
		@rm -f $(GRAFDEP) $(GRAFDS) $(GRAFDH) $(GRAFLIB)

distclean::     distclean-graf
