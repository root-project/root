# Module.mk for netx module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: G. Ganis, 8/7/2004

MODNAME      := netx
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

NETXDIR      := $(MODDIR)
NETXDIRS     := $(NETXDIR)/src
NETXDIRI     := $(NETXDIR)/inc

##### libNetx #####
NETXL        := $(MODDIRI)/LinkDef.h
NETXDS       := $(call stripsrc,$(MODDIRS)/G__Netx.cxx)
NETXDO       := $(NETXDS:.cxx=.o)
NETXDH       := $(NETXDS:.cxx=.h)

NETXH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETXS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
NETXO        := $(call stripsrc,$(NETXS:.cxx=.o))

NETXDEP      := $(NETXO:.o=.d) $(NETXDO:.o=.d)

NETXLIB      := $(LPATH)/libNetx.$(SOEXT)
NETXMAP      := $(NETXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETXH))
ALLLIBS      += $(NETXLIB)
ALLMAPS      += $(NETXMAP)

# include all dependency files
INCLUDEFILES += $(NETXDEP)

# When using an external XROOTD distribution XROOTDDIRI and XROOTDDIRL
# are undefined and have to point to the specified inc and lib dirs.
ifneq ($(XRDINCDIR),)
ifeq ($(XROOTDDIRI),)
XROOTDDIRI   := $(XRDINCDIR)
endif
endif
ifneq ($(XRDLIBDIR),)
ifeq ($(XROOTDDIRL),)
XROOTDDIRL   := $(XRDLIBDIR)
endif
endif

# Xrootd includes
NETXINCEXTRA := $(XROOTDDIRI:%=-I%)
ifneq ($(EXTRA_XRDFLAGS),)
NETXINCEXTRA += -Iproof/proofd/inc
endif

# Xrootd client libs
ifeq ($(PLATFORM),win32)
NETXLIBEXTRA += $(XROOTDDIRL)/libXrdClient.lib
else
NETXLIBEXTRA += -L$(XROOTDDIRL) -lXrdOuc -lXrdSys \
                -lXrdClient
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(NETXDIRI)/%.h $(XROOTDMAKE)
		cp $< $@

$(NETXLIB):     $(NETXO) $(NETXDO) $(ORDER_) $(MAINLIBS) $(NETXLIBDEP) \
                $(XRDNETXD)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNetx.$(SOEXT) $@ "$(NETXO) $(NETXDO)" \
		   "$(NETXLIBEXTRA)"

$(NETXDS):      $(NETXH1) $(NETXL) $(XROOTDMAKE) $(ROOTCINTTMPDEP) $(XRDPLUGINS)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(NETXINCEXTRA) $(NETXH) $(NETXL)

$(NETXMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(NETXL)
		$(RLIBMAP) -o $@ -l $(NETXLIB) -d $(NETXLIBDEPM) -c $(NETXL)

all-$(MODNAME): $(NETXLIB) $(NETXMAP)

clean-$(MODNAME):
		@rm -f $(NETXO) $(NETXDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(NETXDEP) $(NETXDS) $(NETXDH) $(NETXLIB) $(NETXMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(NETXO) $(NETXDO): $(XROOTDMAKE) $(XRDHDRS)
$(NETXO) $(NETXDO): CXXFLAGS += $(NETXINCEXTRA) $(EXTRA_XRDFLAGS)
ifeq ($(PLATFORM),win32)
$(NETXO) $(NETXDO): CXXFLAGS += -DNOGDI $(NETXINCEXTRA) $(EXTRA_XRDFLAGS)
endif
