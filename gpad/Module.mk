# Module.mk for gpad module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := gpad
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GPADDIR      := $(MODDIR)
GPADDIRS     := $(GPADDIR)/src
GPADDIRI     := $(GPADDIR)/inc

##### libGpad #####
GPADL        := $(MODDIRI)/LinkDef.h
GPADDS       := $(MODDIRS)/G__GPad.cxx
GPADDO       := $(GPADDS:.cxx=.o)
GPADDH       := $(GPADDS:.cxx=.h)

GPADH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GPADS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GPADO        := $(GPADS:.cxx=.o)

GPADDEP      := $(GPADO:.o=.d) $(GPADDO:.o=.d)

GPADLIB      := $(LPATH)/libGpad.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GPADH))
ALLLIBS     += $(GPADLIB)

# include all dependency files
INCLUDEFILES += $(GPADDEP)

##### local rules #####
include/%.h:    $(GPADDIRI)/%.h
		cp $< $@

$(GPADLIB):     $(GPADO) $(GPADDO) $(MAINLIBS) $(GPADLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGpad.$(SOEXT) $@ "$(GPADO) $(GPADDO)" \
		   "$(GPADLIBEXTRA)"

$(GPADDS):      $(GPADH) $(GPADL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GPADH) $(GPADL)

$(GPADDO):      $(GPADDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-gpad:       $(GPADLIB)

map-gpad:       $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GPADLIB) \
		   -d $(GPADLIBDEP) -c $(GPADL)

map::           map-gpad

clean-gpad:
		@rm -f $(GPADO) $(GPADDO)

clean::         clean-gpad

distclean-gpad: clean-gpad
		@rm -f $(GPADDEP) $(GPADDS) $(GPADDH) $(GPADLIB)

distclean::     distclean-gpad
