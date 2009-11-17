# Module.mk for proofd module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := proofd
MODDIR       := proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFDDIR    := $(MODDIR)
PROOFDDIRS   := $(PROOFDDIR)/src
PROOFDDIRI   := $(PROOFDDIR)/inc

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

ifeq ($(PLATFORM),win32)

##### XrdProofd plugin ####
XPDH         := $(wildcard $(MODDIRI)/X*.h)
XPDS         := $(wildcard $(MODDIRS)/X*.cxx)
XPDO         := $(XPDS:.cxx=.o)
XPDO         := $(MODDIRS)/XProofProtUtils.o


##### Object files used by libProofx #####
XPCONNH      := $(MODDIRI)/XrdProofConn.h $(MODDIRI)/XrdProofPhyConn.h \
                $(MODDIRI)/XProofProtUtils.h

XPCONNS      := $(MODDIRS)/XrdProofConn.cxx $(MODDIRS)/XrdProofPhyConn.cxx \
                $(MODDIRS)/XProofProtUtils.cxx

XPCONNO      := $(MODDIRS)/XrdProofConn.o $(MODDIRS)/XrdProofPhyConn.o \
                $(MODDIRS)/XProofProtUtils.o

XPDDEP       := $(XPCONNO:.o=.d)

XPDLIB       := $(LPATH)/libXrdProofd.$(SOEXT)

# Extra include paths and libs
XPDINCEXTRA    := $(XROOTDDIRI:%=-I%)
XPDINCEXTRA    += $(PROOFDDIRI:%=-I%)

XPDLIBEXTRA  += $(XROOTDDIRL)/libXrdClient.lib

# used in the main Makefile
PROOFDEXEH   := $(MODDIRI)/proofdp.h
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFDEXEH))
ALLLIBS      += $(XPDLIB)

# include all dependency files
INCLUDEFILES += $(XPDDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PROOFDDIRI)/%.h
		cp $< $@

$(XPDLIB):      $(XPCONNO) $(XPCONNH) $(XRDPLUGINS) $(ORDER_) $(MAINLIBS)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXrdProofd.$(SOEXT) $@ "$(XPDO)" \
		   "$(XPDLIBEXTRA)"

all-$(MODNAME): $(XPDLIB)

clean-$(MODNAME):
		@rm -f $(XPCONNO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(XPDDEP) $(XPDLIB)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(XPCONNO): CXXFLAGS += $(XPDINCEXTRA) $(EXTRA_XRDFLAGS)

else

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

# Extra include paths and libs
XPROOFDEXELIBS :=
XPROOFDEXE     :=
ifeq ($(BUILDXRD),yes)
XPDINCEXTRA    := $(XROOTDDIRI:%=-I%)
XPDINCEXTRA    += $(PROOFDDIRI:%=-I%)
XPDLIBEXTRA    += -L$(XROOTDDIRL) -lXrdOuc -lXrdNet -lXrdSys \
                  -lXrdClient -lXrdSut
XPROOFDEXELIBS := $(XROOTDDIRL)/libXrd.a $(XROOTDDIRL)/libXrdClient.a \
                  $(XROOTDDIRL)/libXrdNet.a $(XROOTDDIRL)/libXrdOuc.a \
                  $(XROOTDDIRL)/libXrdSys.a $(XROOTDDIRL)/libXrdSut.a
ifeq ($(PLATFORM),solaris)
XPROOFDEXELIBS += -lsendfile
endif
XPROOFDEXE     := bin/xproofd
endif

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFDEXEH))
ALLEXECS     += $(PROOFDEXE)
ifeq ($(BUILDXRD),yes)
ALLLIBS      += $(XPDLIB)
ALLEXECS     += $(XPROOFDEXE)
endif

# include all dependency files
ifeq ($(BUILDXRD),yes)
INCLUDEFILES += $(PROOFDDEP) $(XPDDEP)
else
INCLUDEFILES += $(PROOFDDEP)
endif

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PROOFDDIRI)/%.h
		cp $< $@

$(PROOFDEXE):   $(PROOFDEXEO) $(RSAO) $(SNPRINTFO) $(GLBPATCHO) $(RPDUTILO)
		$(LD) $(LDFLAGS) -o $@ $(PROOFDEXEO) $(RPDUTILO) $(GLBPATCHO) \
		   $(RSAO) $(SNPRINTFO) $(CRYPTLIBS) $(AUTHLIBS) $(SYSLIBS)

$(XPROOFDEXE):  $(XPDO) $(XPROOFDEXELIBS) $(XRDPROOFXD)
		$(LD) $(LDFLAGS) -o $@ $(XPDO) $(XPROOFDEXELIBS) $(SYSLIBS)

$(XPDLIB):      $(XPDO) $(XPDH) $(ORDER_) $(MAINLIBS) $(XRDPROOFXD)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libXrdProofd.$(SOEXT) $@ "$(XPDO)" \
		   "$(XPDLIBEXTRA)"

all-$(MODNAME): $(PROOFDEXE) $(XPROOFDEXE) $(XPDLIB)

clean-$(MODNAME):
		@rm -f $(PROOFDEXEO) $(XPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PROOFDDEP) $(PROOFDEXE) $(XPROOFDEXE) $(XPDDEP) $(XPDLIB)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(PROOFDEXEO): CXXFLAGS += $(AUTHFLAGS)
$(PROOFDEXEO): PCHCXXFLAGS =
$(XPDO):       PCHCXXFLAGS =

$(XPDO): $(XRDHDRS)
ifneq ($(ICC_GE_9),)
# remove when xrootd has moved from strstream.h -> sstream.
$(XPDO): CXXFLAGS += -Wno-deprecated $(XPDINCEXTRA) $(EXTRA_XRDFLAGS)
else
ifneq ($(GCC_MAJOR),)
ifneq ($(GCC_MAJOR),2)
# remove when xrootd has moved from strstream.h -> sstream.
$(XPDO): CXXFLAGS += -Wno-deprecated $(XPDINCEXTRA) $(EXTRA_XRDFLAGS)
else
$(XPDO): CXXFLAGS += $(XPDINCEXTRA) $(EXTRA_XRDFLAGS)
endif
else
$(XPDO): CXXFLAGS += $(XPDINCEXTRA) $(EXTRA_XRDFLAGS)
endif
endif
endif
