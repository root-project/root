# Module.mk for net module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := net
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

NETDIR       := $(MODDIR)
NETDIRS      := $(NETDIR)/src
NETDIRI      := $(NETDIR)/inc

##### libNet (part of libCore) #####
NETL         := $(MODDIRI)/LinkDef.h
NETDS        := $(MODDIRS)/G__Net.cxx
NETDO        := $(NETDS:.cxx=.o)
NETDH        := $(NETDS:.cxx=.h)

NETH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
NETO         := $(NETS:.cxx=.o)

NETDEP       := $(NETO:.o=.d) $(NETDO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETH))

# include all dependency files
INCLUDEFILES += $(NETDEP)

##### local rules #####
include/%.h:    $(NETDIRI)/%.h
		cp $< $@

$(NETDS):       $(NETH) $(NETL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(NETH) $(NETL)

$(NETDO):       $(NETDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-net:        $(NETO) $(NETDO)

clean-net:
		@rm -f $(NETO) $(NETDO)

clean::         clean-net

distclean-net:  clean-net
		@rm -f $(NETDEP) $(NETDS) $(NETDH)

distclean::     distclean-net
