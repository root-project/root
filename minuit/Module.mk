# Module.mk for minuit module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := minuit
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MINUITDIR    := $(MODDIR)
MINUITDIRS   := $(MINUITDIR)/src
MINUITDIRI   := $(MINUITDIR)/inc

##### libMinuit #####
MINUITL      := $(MODDIRI)/LinkDef.h
MINUITDS     := $(MODDIRS)/G__Minuit.cxx
MINUITDO     := $(MINUITDS:.cxx=.o)
MINUITDH     := $(MINUITDS:.cxx=.h)

MINUITH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MINUITS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MINUITO      := $(MINUITS:.cxx=.o)

MINUITDEP    := $(MINUITO:.o=.d) $(MINUITDO:.o=.d)

MINUITLIB    := $(LPATH)/libMinuit.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MINUITH))
ALLLIBS     += $(MINUITLIB)

# include all dependency files
INCLUDEFILES += $(MINUITDEP)

##### local rules #####
include/%.h:    $(MINUITDIRI)/%.h
		cp $< $@

$(MINUITLIB):   $(MINUITO) $(MINUITDO) $(MAINLIBS) $(MINUITLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMinuit.$(SOEXT) $@ "$(MINUITO) $(MINUITDO)" \
		   "$(MINUITLIBEXTRA)"

$(MINUITDS):    $(MINUITH) $(MINUITL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MINUITH) $(MINUITL)

$(MINUITDO):    $(MINUITDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-minuit:     $(MINUITLIB)

map-minuit:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(MINUITLIB) \
		   -d $(MINUITLIBDEP) -c $(MINUITL)

map::           map-minuit

clean-minuit:
		@rm -f $(MINUITO) $(MINUITDO)

clean::         clean-minuit

distclean-minuit: clean-minuit
		@rm -f $(MINUITDEP) $(MINUITDS) $(MINUITDH) $(MINUITLIB)

distclean::     distclean-minuit
