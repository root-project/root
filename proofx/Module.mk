# Module.mk for the proofx module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Gerardo Ganis  12/12/2005

MODDIR       := proofx
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFXDIR    := $(MODDIR)
PROOFXDIRS   := $(PROOFXDIR)/src
PROOFXDIRI   := $(PROOFXDIR)/inc

##### libProofx #####
PROOFXL      := $(MODDIRI)/LinkDef.h
PROOFXDS     := $(MODDIRS)/G__Proofx.cxx
PROOFXDO     := $(PROOFXDS:.cxx=.o)
PROOFXDH     := $(PROOFXDS:.cxx=.h)

ifeq ($(PLATFORM),win32)
PROOFXH      := $(MODDIRI)/TXProofMgr.h $(MODDIRI)/TXSlave.h \
                $(MODDIRI)/TXSocket.h $(MODDIRI)/TXSocketHandler.h
PROOFXS      := $(MODDIRS)/TXProofMgr.cxx $(MODDIRS)/TXSlave.cxx \
                $(MODDIRS)/TXSocket.cxx $(MODDIRS)/TXSocketHandler.cxx
else
PROOFXH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PROOFXS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
endif
PROOFXO      := $(PROOFXS:.cxx=.o)

PROOFXDEP    := $(PROOFXO:.o=.d) $(PROOFXDO:.o=.d)

PROOFXLIB    := $(LPATH)/libProofx.$(SOEXT)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFXH))
ALLLIBS      += $(PROOFXLIB)

# include all dependency files
INCLUDEFILES += $(PROOFXDEP)

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
endif
endif

# Xrootd includes
PROOFXINCEXTRA := $(PROOFXDIRI:%=-I%)
PROOFXINCEXTRA += $(XROOTDDIRI:%=-I%)
PROOFXINCEXTRA += $(PROOFDDIRI:%=-I%)

# Xrootd client libs
ifeq ($(PLATFORM),win32)
PROOFXLIBEXTRA := $(XROOTDDIRL)/libXrdClient.lib
PROOFXLIBEXTRA += $(ROOTSYS)/lib/libThread.lib $(ROOTSYS)/lib/libProof.lib
else
PROOFXLIBEXTRA := $(XROOTDDIRL)/libXrdClient.a $(XROOTDDIRL)/libXrdOuc.a \
		  $(XROOTDDIRL)/libXrdNet.a
endif

##### local rules #####
include/%.h:    $(PROOFXDIRI)/%.h
		cp $< $@

$(PROOFXLIB):   $(PROOFXO) $(XPCONNO) $(PROOFXDO) $(ORDER_) $(MAINLIBS) \
                $(PROOFXLIBDEP) $(XRDPLUGINS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofx.$(SOEXT) $@ \
		   "$(PROOFXO) $(XPCONNO) $(PROOFXDO)" \
		   "$(PROOFXLIBEXTRA)"

$(PROOFXDS):    $(PROOFXH1) $(PROOFXL) $(ROOTCINTTMPEXE) $(XROOTDETAG)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PROOFXINCEXTRA) $(PROOFXH) $(PROOFXL)

all-proofx:     $(PROOFXLIB)

map-proofx:     $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PROOFXLIB) \
		   -d $(PROOFXLIBDEP) -c $(PROOFXL)

map::           map-proofx

clean-proofx:
		@rm -f $(PROOFXO) $(PROOFXDO)

clean::         clean-proofx

distclean-proofx: clean-proofx
		@rm -f $(PROOFXDEP) $(PROOFXDS) $(PROOFXDH) $(PROOFXLIB)

distclean::     distclean-proofx

##### extra rules ######
$(PROOFXO) $(PROOFXDO): $(XROOTDETAG)

ifeq ($(ICC_MAJOR),9)
# remove when xrootd has moved from strstream.h -> sstream.
$(PROOFXO) $(PROOFXDO): CXXFLAGS += -Wno-deprecated $(PROOFXINCEXTRA)
else
ifeq ($(GCC_MAJOR),4)
# remove when xrootd has moved from strstream.h -> sstream.
$(PROOFXO) $(PROOFXDO): CXXFLAGS += -Wno-deprecated $(PROOFXINCEXTRA)
else
$(PROOFXO) $(PROOFXDO): CXXFLAGS += $(PROOFXINCEXTRA)
endif
endif
