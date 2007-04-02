# Module.mk for netx module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G. Ganis, 8/7/2004

MODDIR       := netx
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

NETXDIR      := $(MODDIR)
NETXDIRS     := $(NETXDIR)/src
NETXDIRI     := $(NETXDIR)/inc

##### libNetx #####
NETXL        := $(MODDIRI)/LinkDef.h
NETXDS       := $(MODDIRS)/G__Netx.cxx
NETXDO       := $(NETXDS:.cxx=.o)
NETXDH       := $(NETXDS:.cxx=.h)

NETXH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETXS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
NETXO        := $(NETXS:.cxx=.o)

NETXDEP      := $(NETXO:.o=.d) $(NETXDO:.o=.d)

NETXLIB      := $(LPATH)/libNetx.$(SOEXT)
NETXMAP      := $(NETXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETXH))
ALLLIBS      += $(NETXLIB)
ALLMAPS      += $(NETXMAP)

# include all dependency files
INCLUDEFILES += $(NETXDEP)

# These are undefined if using an external XROOTD distribution
# The new XROOTD build system based on autotools installs the headers
# under <dir>/include/xrootd, while the old system under <dir>/src
ifneq ($(XROOTDDIR),)
ifeq ($(XROOTDDIRI),)
XROOTDDIRI   := $(XROOTDDIR)/include/xrootd
ifeq ($(wildcard $(XROOTDDIRI)/*.hh),)
XROOTDDIRI   := $(XROOTDDIR)/src
endif
XROOTDDIRL   := $(XROOTDDIR)/lib
XROOTDDIRP   := $(XROOTDDIRL)
endif
endif

# Xrootd includes
NETXINCEXTRA := $(XROOTDDIRI:%=-I%)

# Xrootd client libs
ifeq ($(PLATFORM),win32)
NETXLIBEXTRA += $(XROOTDDIRL)/libXrdClient.lib
else
NETXLIBEXTRA += -L$(XROOTDDIRP) -lXrdPosix
endif

##### local rules #####
include/%.h:    $(NETXDIRI)/%.h $(XROOTDETAG)
		cp $< $@

$(NETXLIB):     $(NETXO) $(NETXDO) $(ORDER_) $(MAINLIBS) $(NETXLIBDEP) \
                $(XRDPLUGINS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNetx.$(SOEXT) $@ "$(NETXO) $(NETXDO)" \
		   "$(NETXLIBEXTRA)"

$(NETXDS):      $(NETXH1) $(NETXL) $(ROOTCINTTMPEXE) $(XROOTDETAG)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(NETXINCEXTRA) $(NETXH) $(NETXL)

$(NETXMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(NETXL)
		$(RLIBMAP) -o $(NETXMAP) -l $(NETXLIB) -d $(NETXLIBDEPM) -c $(NETXL)

all-netx:       $(NETXLIB) $(NETXMAP)

clean-netx:
		@rm -f $(NETXO) $(NETXDO)

clean::         clean-netx

distclean-netx: clean-netx
		@rm -f $(NETXDEP) $(NETXDS) $(NETXDH) $(NETXLIB) $(NETXMAP)

distclean::     distclean-netx

##### extra rules ######
$(NETXO) $(NETXDO): $(XROOTDETAG)
$(NETXO) $(NETXDO): CXXFLAGS += $(NETXINCEXTRA)
ifeq ($(PLATFORM),win32)
$(NETXO) $(NETXDO): CXXFLAGS += -DNOGDI
endif
