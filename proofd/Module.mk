# Module.mk for proofd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := proofd
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFDDIR    := $(MODDIR)
PROOFDDIRS   := $(PROOFDDIR)/src
PROOFDDIRI   := $(PROOFDDIR)/inc

##### proofd #####
PROOFDEXEH   := $(MODDIRI)/proofdp.h
PROOFDEXES   := $(MODDIRS)/proofd.cxx
PROOFDEXEO   := $(PROOFDEXES:.cxx=.o)
PROOFDDEP    := $(PROOFDEXEO:.o=.d)
PROOFDEXE    := bin/proofd

##### XrdProofd plugin ####
XPDH         := $(wildcard $(MODDIRI)/X*.h)
XPDS         := $(wildcard $(MODDIRS)/X*.cxx)
XPDO         := $(XPDS:.cxx=.o)

XPDDEP       := $(XPDO:.o=.d)

XPDLIB       := $(LPATH)/libXrdProofd.$(SOEXT)

##### Object files used by libProofx #####
XPCONNO      := $(MODDIRS)/XrdProofConn.o $(MODDIRS)/XrdProofPhyConn.o \
                $(MODDIRS)/XProofProtUtils.o

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

# Extra include paths and libs
ifeq ($(BUILDXRD),yes)
XPDINCEXTRA    := $(XROOTDDIRI:%=-I%)
XPDINCEXTRA    += $(PROOFDDIRI:%=-I%)
XPDLIBEXTRA    += $(XROOTDDIRL)/libXrdClient.a $(XROOTDDIRL)/libXrdOuc.a \
                  $(XROOTDDIRL)/libXrdNet.a
endif

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFDEXEH))
ALLEXECS     += $(PROOFDEXE)
ifeq ($(BUILDXRD),yes)
ALLLIBS      += $(XPDLIB)
endif

# include all dependency files
ifeq ($(BUILDXRD),yes)
INCLUDEFILES += $(PROOFDDEP) $(XPDDEP)
else
INCLUDEFILES += $(PROOFDDEP)
endif

##### local rules #####
include/%.h:    $(PROOFDDIRI)/%.h
		cp $< $@

$(PROOFDEXE):   $(PROOFDEXEO) $(RSAO) $(SNPRINTFO) $(GLBPATCHO) $(RPDUTILO)
		$(LD) $(LDFLAGS) -o $@ $(PROOFDEXEO) $(RPDUTILO) $(GLBPATCHO) \
		   $(RSAO) $(SNPRINTFO) $(CRYPTLIBS) $(AUTHLIBS) $(SYSLIBS)

$(XPDLIB):      $(XPDO) $(XPDH) $(ORDER_) $(MAINLIBS) $(XRDPLUGINS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXrdProofd.$(SOEXT) $@ "$(XPDO)" \
		   "$(XPDLIBEXTRA)"

all-proofd:     $(PROOFDEXE) $(XPDLIB)

clean-proofd:
		@rm -f $(PROOFDEXEO) $(XPDO)

clean::         clean-proofd

distclean-proofd: clean-proofd
		@rm -f $(PROOFDDEP) $(PROOFDEXE) $(XPDDEP) $(XPDLIB)

distclean::     distclean-proofd

##### extra rules ######
$(PROOFDEXEO): CXXFLAGS += $(AUTHFLAGS)

$(XPDO): $(XRDPLUGINS)
ifeq ($(ICC_MAJOR),9)
# remove when xrootd has moved from strstream.h -> sstream.
$(XPDO): CXXFLAGS += -Wno-deprecated $(XPDINCEXTRA)
else
ifeq ($(GCC_MAJOR),4)
# remove when xrootd has moved from strstream.h -> sstream.
$(XPDO): CXXFLAGS += -Wno-deprecated $(XPDINCEXTRA)
else
$(XPDO): CXXFLAGS += $(XPDINCEXTRA)
endif
endif
