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

#### DaemonUtils goes into libSrvAuth ####
NETH         := $(filter-out $(MODDIRI)/DaemonUtils.h,$(NETH))
NETS         := $(filter-out $(MODDIRS)/DaemonUtils.cxx,$(NETS))
NETO         := $(filter-out $(MODDIRS)/DaemonUtils.o,$(NETO))

DAEMONUTILSO := $(MODDIRS)/DaemonUtils.o

# Add SSL flags, if required
ifneq ($(SSLLIB),)
SSLFLAGS     := $(SSLINCDIR:%=-I%)
ifneq ($(CRYPTLIBS),)
CRYPTLIBS    += $(SSLLIBDIR) $(SSLLIB)
else
CRYPTLIBS     = $(SSLLIBDIR) $(SSLLIB)
endif
endif
EXTRANETFLAGS = $(SSLFLAGS) $(EXTRA_AUTHFLAGS)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETH)) \
                include/DaemonUtils.h

# include all dependency files
INCLUDEFILES += $(NETDEP)

##### local rules #####
include/%.h:    $(NETDIRI)/%.h
		cp $< $@

$(NETDS):       $(NETH) $(NETL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(NETH) $(NETL)

$(NETO):        %.o: %.cxx
		$(CXX) $(OPT) $(CXXFLAGS) $(EXTRANETFLAGS) -I. -o $@ -c $<

$(NETDO):       $(NETDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(EXTRANETFLAGS) -I. -o $@ -c $<

all-net:        $(NETO) $(NETDO)

clean-net:
		@rm -f $(NETO) $(NETDO) $(DAEMONUTILSO)

clean::         clean-net

distclean-net:  clean-net
		@rm -f $(NETDEP) $(NETDS) $(NETDH)

distclean::     distclean-net
