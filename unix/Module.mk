# Module.mk for unix module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := unix
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UNIXDIR      := $(MODDIR)
UNIXDIRS     := $(UNIXDIR)/src
UNIXDIRI     := $(UNIXDIR)/inc

##### libUnix (part of libCore) #####
UNIXL        := $(MODDIRI)/LinkDef.h
UNIXDS       := $(MODDIRS)/G__Unix.cxx
UNIXDO       := $(UNIXDS:.cxx=.o)
UNIXDH       := $(UNIXDS:.cxx=.h)

UNIXH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
UNIXS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
UNIXO        := $(UNIXS:.cxx=.o)

UNIXDEP      := $(UNIXO:.o=.d) $(UNIXDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(UNIXH))

# include all dependency files
INCLUDEFILES += $(UNIXDEP)

##### local rules #####
include/%.h:    $(UNIXDIRI)/%.h
		cp $< $@

$(UNIXDS):      $(UNIXH) $(UNIXL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(UNIXH) $(UNIXL)

$(UNIXDO):      $(UNIXDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-unix:       $(UNIXO) $(UNIXDO)

clean-unix:
		@rm -f $(UNIXO) $(UNIXDO)

clean::         clean-unix

distclean-unix: clean-unix
		@rm -f $(UNIXDEP) $(UNIXDS) $(UNIXDH)

distclean::     distclean-unix
