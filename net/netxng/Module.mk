# Module.mk for xrd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Lukasz Janyst 11/01/2013

MODNAME        := netxng
MODDIR         := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

NETXNGDIR      := $(MODDIR)
NETXNGDIRS     := $(MODDIR)/src
NETXNGDIRI     := $(MODDIR)/inc

##### libNetxNG #####
NETXNGL        := $(MODDIRI)/LinkDef.h
NETXNGDS       := $(call stripsrc,$(MODDIRS)/G__NetxNG.cxx)
NETXNGDO       := $(NETXNGDS:.cxx=.o)
NETXNGDH       := $(NETXNGDS:.cxx=.h)

NETXNGH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETXNGS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
NETXNGO        := $(call stripsrc,$(NETXNGS:.cxx=.o))

NETXNGDEP      := $(NETXNGO:.o=.d) $(NETXNGDO:.o=.d)

NETXNGLIB      := $(LPATH)/libNetxNG.$(SOEXT)
NETXNGMAP      := $(NETXNGLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS        += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETXNGH))
ALLLIBS        += $(NETXNGLIB)
ALLMAPS        += $(NETXNGMAP)

# include all dependency files
INCLUDEFILES   += $(NETXNGDEP)

# Xrootd includes
NETXNGINCEXTRA := $(XRDINCDIR:%=-I%)

# Xrootd client libs
ifeq ($(PLATFORM),win32)
NETXNGLIBEXTRA += $(XRDLIBDIR)/libXrdCl.lib
else
NETXNGLIBEXTRA += $(XRDLIBDIR) -lXrdUtils -lXrdCl
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(NETXNGDIRI)/%.h
		cp $< $@

$(NETXNGLIB):   $(NETXNGO) $(NETXNGDO) $(ORDER_) $(MAINLIBS) $(NETXNGLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNetxNG.$(SOEXT) $@ "$(NETXNGO) $(NETXNGDO)" \
		   "$(NETXNGLIBEXTRA)"

$(call pcmrule,NETXNG)
	$(noop)

$(NETXNGDS):    $(NETXNGH) $(NETXNGL) $(XROOTDMAKE) $(ROOTCLINGEXE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,NETXNG) -c $(NETXNGINCEXTRA) $(NETXNGH) $(NETXNGL)

$(NETXNGMAP):   $(NETXNGH) $(NETXNGL) $(XROOTDMAKE) $(ROOTCLINGEXE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(NETXNGDS) $(call dictModule,NETXNG) -c $(NETXNGINCEXTRA) $(NETXNGH) $(NETXNGL)

all-$(MODNAME): $(NETXNGLIB)

clean-$(MODNAME):
		@rm -f $(NETXNGO) $(NETXNGDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(NETXNGDEP) $(NETXNGDS) $(NETXNGDH) $(NETXNGLIB) $(NETXNGMAP)

distclean::     distclean-$(MODNAME)

$(NETXNGO) $(NETXNGDO): CXXFLAGS += $(NETXNGINCEXTRA)
ifneq ($(findstring gnu,$(COMPILER)),)
# problem in xrootd 3.3.5 headers
$(NETXNGO) $(NETXNGDO): CXXFLAGS += -Wno-unused-parameter -Wno-shadow
endif
