# Module.mk for hist module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := hist
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HISTDIR      := $(MODDIR)
HISTDIRS     := $(HISTDIR)/src
HISTDIRI     := $(HISTDIR)/inc

##### libHist #####
HISTL        := $(MODDIRI)/LinkDef.h
HISTDS       := $(MODDIRS)/G__Hist.cxx
HISTDO       := $(HISTDS:.cxx=.o)
HISTDH       := $(HISTDS:.cxx=.h)

HISTH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HISTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HISTO        := $(HISTS:.cxx=.o)

HISTDEP      := $(HISTO:.o=.d) $(HISTDO:.o=.d)

HISTLIB      := $(LPATH)/libHist.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(HISTH))
ALLLIBS     += $(HISTLIB)

# include all dependency files
INCLUDEFILES += $(HISTDEP)

##### local rules #####
include/%.h:    $(HISTDIRI)/%.h
		cp $< $@

$(HISTLIB):     $(HISTO) $(HISTDO) $(MAINLIBS) $(HISTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHist.$(SOEXT) $@ "$(HISTO) $(HISTDO)" \
		   "$(HISTLIBEXTRA)"

$(HISTDS):      $(HISTH) $(HISTL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(HISTH) $(HISTL)

$(HISTDO):      $(HISTDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-hist:       $(HISTLIB)

map-hist:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(HISTLIB) \
		   -d $(HISTLIBDEP) -c $(HISTL)

map::           map-hist

clean-hist:
		@rm -f $(HISTO) $(HISTDO)

clean::         clean-hist

distclean-hist: clean-hist
		@rm -f $(HISTDEP) $(HISTDS) $(HISTDH) $(HISTLIB)

distclean::     distclean-hist
