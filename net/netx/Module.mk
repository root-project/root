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

ifeq ($(HASXRD),yes)
# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETXH))
ALLLIBS      += $(NETXLIB)
ALLMAPS      += $(NETXMAP)

# include all dependency files
INCLUDEFILES += $(NETXDEP)
endif

# Xrootd includes
NETXINCEXTRA := $(XRDINCDIR:%=-I%)
ifneq ($(EXTRA_XRDFLAGS),)
NETXINCEXTRA += -I$(ROOT_SRCDIR)/proof/proofd/inc
endif
ifeq ($(XRDINCPRIVATE),yes)
NETXINCEXTRA += -I$(XRDINCDIR)/private
else
ifeq ($(XRDINCPRIVATE),proof)
NETXINCEXTRA += -Iproof/xrdinc
endif
endif

# Xrootd client libs
ifeq ($(PLATFORM),win32)
NETXLIBEXTRA += $(XRDLIBDIR)/libXrdClient.lib
else
ifeq ($(HASXRDUTILS),no)
NETXLIBEXTRA += $(XRDLIBDIR) -lXrdOuc -lXrdSys -lXrdClient -lpthread
else
NETXLIBEXTRA += $(XRDLIBDIR) -lXrdUtils -lXrdClient
endif
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(NETXDIRI)/%.h $(XROOTDMAKE)
		cp $< $@

$(NETXLIB):     $(NETXO) $(NETXDO) $(ORDER_) $(MAINLIBS) $(NETXLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNetx.$(SOEXT) $@ "$(NETXO) $(NETXDO)" \
		   "$(NETXLIBEXTRA)"

$(call pcmrule,NETX)
	$(noop)

$(NETXDS):      $(NETXH) $(NETXL) $(XROOTDMAKE) $(ROOTCLINGEXE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,NETX) -c $(NETXINCEXTRA) $(NETXH) $(NETXL)

$(NETXMAP):     $(NETXH) $(NETXL) $(XROOTDMAKE) $(ROOTCLINGEXE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(NETXDS) $(call dictModule,NETX) -c $(NETXINCEXTRA) $(NETXH) $(NETXL)

all-$(MODNAME): $(NETXLIB)

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
