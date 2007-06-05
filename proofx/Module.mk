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
                $(MODDIRI)/TXSocket.h $(MODDIRI)/TXSocketHandler.h \
                $(MODDIRI)/TXHandler.h
PROOFXS      := $(MODDIRS)/TXProofMgr.cxx $(MODDIRS)/TXSlave.cxx \
                $(MODDIRS)/TXSocket.cxx $(MODDIRS)/TXSocketHandler.cxx \
                $(MODDIRS)/TXHandler.cxx
else
PROOFXH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PROOFXS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
endif
PROOFXO      := $(PROOFXS:.cxx=.o)

PROOFXDEP    := $(PROOFXO:.o=.d) $(PROOFXDO:.o=.d)

PROOFXLIB    := $(LPATH)/libProofx.$(SOEXT)
PROOFXMAP    := $(PROOFXLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFXH))
ALLLIBS      += $(PROOFXLIB)
ALLMAPS      += $(PROOFXMAP)

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
XROOTDDIRP   := $(XROOTDDIRL)
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
PROOFXLIBEXTRA += -L$(XROOTDDIRP) -lXrdClient \
                  $(XROOTDDIRL)/libXrdOuc.a $(XROOTDDIRL)/libXrdNet.a
endif

##### local rules #####
include/%.h:    $(PROOFXDIRI)/%.h $(XROOTDETAG)
		cp $< $@

$(PROOFXLIB):   $(PROOFXO) $(PROOFXDO) $(XPCONNO) $(ORDER_) $(MAINLIBS) \
                $(PROOFXLIBDEP) $(XRDPLUGINS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofx.$(SOEXT) $@ \
		   "$(PROOFXO) $(XPCONNO) $(PROOFXDO)" \
		   "$(PROOFXLIBEXTRA)"

$(PROOFXDS):    $(PROOFXH) $(PROOFXL) $(ROOTCINTTMPEXE) $(XROOTDETAG)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PROOFXINCEXTRA) $(PROOFXH) $(PROOFXL)

$(PROOFXMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(PROOFXL)
		$(RLIBMAP) -o $(PROOFXMAP) -l $(PROOFXLIB) \
		   -d $(PROOFXLIBDEPM) -c $(PROOFXL)

all-proofx:     $(PROOFXLIB) $(PROOFXMAP)

clean-proofx:
		@rm -f $(PROOFXO) $(PROOFXDO)

clean::         clean-proofx

distclean-proofx: clean-proofx
		@rm -f $(PROOFXDEP) $(PROOFXDS) $(PROOFXDH) $(PROOFXLIB) $(PROOFXMAP)

distclean::     distclean-proofx

##### extra rules ######
$(PROOFXO) $(PROOFXDO): $(XROOTDETAG)

ifeq ($(PLATFORM),win32)
$(PROOFXO) $(PROOFXDO): CXXFLAGS += $(PROOFXINCEXTRA)
else
ifneq ($(ICC_GE_9),)
# remove when xrootd has moved from strstream.h -> sstream.
$(PROOFXO) $(PROOFXDO): CXXFLAGS += -Wno-deprecated $(PROOFXINCEXTRA)
else
ifneq ($(GCC_MAJOR),2)
# remove when xrootd has moved from strstream.h -> sstream.
$(PROOFXO) $(PROOFXDO): CXXFLAGS += -Wno-deprecated $(PROOFXINCEXTRA)
else
$(PROOFXO) $(PROOFXDO): CXXFLAGS += $(PROOFXINCEXTRA)
endif
endif
endif
