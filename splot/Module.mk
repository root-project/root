# Module.mk for splot module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 27/8/2003

MODDIR      := splot
MODDIRS     := $(MODDIR)/src
MODDIRI     := $(MODDIR)/inc

SPLOTDIR    := $(MODDIR)
SPLOTDIRS   := $(SPLOTDIR)/src
SPLOTDIRI   := $(SPLOTDIR)/inc

##### libSPlot #####
SPLOTL      := $(MODDIRI)/LinkDef.h
SPLOTDS     := $(MODDIRS)/G__SPlot.cxx
SPLOTDO     := $(SPLOTDS:.cxx=.o)
SPLOTDH     := $(SPLOTDS:.cxx=.h)

SPLOTH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SPLOTS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SPLOTO      := $(SPLOTS:.cxx=.o)

SPLOTDEP    := $(SPLOTO:.o=.d) $(SPLOTDO:.o=.d)

SPLOTLIB    := $(LPATH)/libSPlot.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SPLOTH))
ALLLIBS     += $(SPLOTLIB)

# include all dependency files
INCLUDEFILES += $(SPLOTDEP)

##### local rules #####
include/%.h:    $(SPLOTDIRI)/%.h
		cp $< $@

$(SPLOTLIB):   $(SPLOTO) $(SPLOTDO) $(ORDER_) $(MAINLIBS) $(SPLOTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSPlot.$(SOEXT) $@ "$(SPLOTO) $(SPLOTDO)" \
		   "$(SPLOTLIBEXTRA)"

$(SPLOTDS):    $(SPLOTH) $(SPLOTL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(SPLOTH) $(SPLOTL)

$(SPLOTDO):    $(SPLOTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-splot:     $(SPLOTLIB)

map-splot:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(SPLOTLIB) \
		   -d $(SPLOTLIBDEP) -c $(SPLOTL)

map::           map-splot

clean-splot:
		@rm -f $(SPLOTO) $(SPLOTDO)

clean::         clean-splot

distclean-splot: clean-splot
		@rm -f $(SPLOTDEP) $(SPLOTDS) $(SPLOTDH) $(SPLOTLIB)

distclean::     distclean-splot
