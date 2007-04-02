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

##### libNet #####
NETL         := $(MODDIRI)/LinkDef.h
NETDS        := $(MODDIRS)/G__Net.cxx
NETDO        := $(NETDS:.cxx=.o)
NETDH        := $(NETDS:.cxx=.h)

NETH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
NETO         := $(NETS:.cxx=.o)

NETDEP       := $(NETO:.o=.d) $(NETDO:.o=.d)

NETLIB       := $(LPATH)/libNet.$(SOEXT)
NETMAP       := $(NETLIB:.$(SOEXT)=.rootmap)

EXTRANETFLAGS =

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETH))
ALLLIBS      += $(NETLIB)
ALLMAPS      += $(NETMAP)

# include all dependency files
INCLUDEFILES += $(NETDEP)

##### local rules #####
include/%.h:    $(NETDIRI)/%.h
		cp $< $@

$(NETLIB):      $(NETO) $(NETDO) $(ORDER_) $(MAINLIBS) $(NETLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNet.$(SOEXT) $@ "$(NETO) $(NETDO)" \
		   "$(NETLIBEXTRA)"

$(NETDS):       $(NETH) $(NETL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(NETH) $(NETL)

$(NETMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(NETL)
		$(RLIBMAP) -o $(NETMAP) -l $(NETLIB) -d $(NETLIBDEPM) -c $(NETL)

all-net:        $(NETLIB) $(NETMAP)

clean-net:
		@rm -f $(NETO) $(NETDO)

clean::         clean-net

distclean-net:  clean-net
		@rm -f $(NETDEP) $(NETDS) $(NETDH) $(NETLIB) $(NETMAP)

distclean::     distclean-net
