# Module.mk for the proofx module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Gerardo Ganis  12/12/2005

MODNAME      := proofx
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFXDIR    := $(MODDIR)
PROOFXDIRS   := $(PROOFXDIR)/src
PROOFXDIRI   := $(PROOFXDIR)/inc

##### libProofx #####
PROOFXL      := $(MODDIRI)/LinkDef.h
PROOFXDS     := $(call stripsrc,$(MODDIRS)/G__Proofx.cxx)
PROOFXDO     := $(PROOFXDS:.cxx=.o)
PROOFXDH     := $(PROOFXDS:.cxx=.h)

ifeq ($(PLATFORM),win32)
PROOFXH      := $(MODDIRI)/TXProofMgr.h $(MODDIRI)/TXSlave.h \
                $(MODDIRI)/TXSocket.h $(MODDIRI)/TXSocketHandler.h \
                $(MODDIRI)/TXHandler.h
PROOFXS      := $(MODDIRS)/TXProofMgr.cxx $(MODDIRS)/TXSlave.cxx \
                $(MODDIRS)/TXSocket.cxx $(MODDIRS)/TXSocketHandler.cxx \
                $(MODDIRS)/TXHandler.cxx
else
PROOFXH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PROOFXS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
endif
PROOFXO      := $(call stripsrc,$(PROOFXS:.cxx=.o))

PROOFXDEP    := $(PROOFXO:.o=.d) $(PROOFXDO:.o=.d)

PROOFXLIB    := $(LPATH)/libProofx.$(SOEXT)
PROOFXMAP    := $(PROOFXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFXH))
ALLLIBS      += $(PROOFXLIB)
ALLMAPS      += $(PROOFXMAP)

# include all dependency files
INCLUDEFILES += $(PROOFXDEP)

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
PROOFXINCEXTRA := $(PROOFXDIRI:%=-I%)
PROOFXINCEXTRA += $(XROOTDDIRI:%=-I%)
PROOFXINCEXTRA += $(PROOFDDIRI:%=-I%)

# Xrootd client libs
ifeq ($(PLATFORM),win32)
PROOFXLIBEXTRA += $(XROOTDDIRL)/libXrdClient.lib
else
ifeq ($(HASXRDUTILS),no)
PROOFXLIBEXTRA += -L$(XROOTDDIRL) -lXrdOuc -lXrdSys -lXrdNet -lXrdClient \
                  -lpthread
# Starting from Jul 2010 XrdNet has been split in two libs:
#    XrdNet and XrdNetUtil
# both are needed
XRDNETUTIL     :=
ifneq ($(XRDVERSION),)
XRDNETUTIL     := $(shell if test $(XRDVERSION) -gt 20100729; then \
                             echo "yes"; \
                          fi)
endif
ifeq ($(XRDNETUTIL),yes)
PROOFXLIBEXTRA += -lXrdNetUtil
endif
else
PROOFXLIBEXTRA += -L$(XROOTDDIRL) -lXrdUtils -lXrdClient
endif
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PROOFXDIRI)/%.h $(XROOTDMAKE)
		cp $< $@

$(PROOFXLIB):   $(PROOFXO) $(PROOFXDO) $(XPCONNO) $(ORDER_) $(MAINLIBS) \
                $(PROOFXLIBDEP) $(XRDPROOFXD)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofx.$(SOEXT) $@ \
		   "$(PROOFXO) $(XPCONNO) $(PROOFXDO)" \
		   "$(PROOFXLIBEXTRA)"

$(PROOFXDS):    $(PROOFXH) $(PROOFXL) $(XROOTDMAKE) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PROOFXINCEXTRA) $(PROOFXH) $(PROOFXL)

$(PROOFXMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(PROOFXL)
		$(RLIBMAP) -o $@ -l $(PROOFXLIB) \
		   -d $(PROOFXLIBDEPM) -c $(PROOFXL)

all-$(MODNAME): $(PROOFXLIB) $(PROOFXMAP)

clean-$(MODNAME):
		@rm -f $(PROOFXO) $(PROOFXDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PROOFXDEP) $(PROOFXDS) $(PROOFXDH) $(PROOFXLIB) $(PROOFXMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PROOFXO) $(PROOFXDO): $(XROOTDMAKE) $(XRDHDRS)

ifeq ($(PLATFORM),win32)
$(PROOFXO) $(PROOFXDO): CXXFLAGS += $(PROOFXINCEXTRA) $(EXTRA_XRDFLAGS)
else
ifneq ($(ICC_GE_9),)
# remove when xrootd has moved from strstream.h -> sstream.
$(PROOFXO) $(PROOFXDO): CXXFLAGS += -Wno-deprecated $(PROOFXINCEXTRA) $(EXTRA_XRDFLAGS)
else
ifneq ($(GCC_MAJOR),)
ifneq ($(GCC_MAJOR),2)
# remove when xrootd has moved from strstream.h -> sstream.
$(PROOFXO) $(PROOFXDO): CXXFLAGS += -Wno-deprecated $(PROOFXINCEXTRA) $(EXTRA_XRDFLAGS)
else
$(PROOFXO) $(PROOFXDO): CXXFLAGS += $(PROOFXINCEXTRA) $(EXTRA_XRDFLAGS)
endif
else
$(PROOFXO) $(PROOFXDO): CXXFLAGS += $(PROOFXINCEXTRA) $(EXTRA_XRDFLAGS)
endif
endif
endif
