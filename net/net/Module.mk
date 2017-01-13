# Module.mk for net module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := net
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

NETDIR       := $(MODDIR)
NETDIRS      := $(NETDIR)/src
NETDIRI      := $(NETDIR)/inc

##### libNet #####
NETL         := $(MODDIRI)/LinkDef.h
NETDS        := $(call stripsrc,$(MODDIRS)/G__Net.cxx)
NETDO        := $(NETDS:.cxx=.o)
NETDH        := $(NETDS:.cxx=.h)

NETMH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ifeq ($(CRYPTOLIB),)
NETNOCRYPTO  := -DR__NO_CRYPTO
NETMH        := $(filter-out $(MODDIRI)/TS3WebFile.h,$(NETMH))
NETMH        := $(filter-out $(MODDIRI)/TS3HTTPRequest.h,$(NETMH))
NETS         := $(filter-out $(MODDIRS)/TS3WebFile.cxx,$(NETS))
NETS         := $(filter-out $(MODDIRS)/TS3HTTPRequest.cxx,$(NETS))
else
NETNOCRYPTO  :=
ifneq (,$(filter $(ARCH),win32gcc win64gcc))
CRYPTOLIB    += -lz
endif
endif

ifeq ($(SSLLIB),)
NETSSL       :=
NETMH        := $(filter-out $(MODDIRI)/TSSLSocket.h,$(NETMH))
NETS         := $(filter-out $(MODDIRS)/TSSLSocket.cxx,$(NETS))
else
NETSSL       := -DR__SSL
endif

NETO         := $(call stripsrc,$(NETS:.cxx=.o))
NETDEP       := $(NETO:.o=.d) $(NETDO:.o=.d)

NETLIB       := $(LPATH)/libNet.$(SOEXT)
NETMAP       := $(NETLIB:.$(SOEXT)=.rootmap)

EXTRANETFLAGS =

NETINCH      := $(subst $(MODDIRI)/%.h,include/%.h,$(NETMH))

# used in the main Makefile
NETMH_REL    := $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETMH))
ALLHDRS      += $(NETMH_REL)
ALLLIBS      += $(NETLIB)
ALLMAPS      += $(NETMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(NETMH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Net_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(NETLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(NETDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(NETDIRI)/%.h
		cp $< $@

$(NETLIB):      $(NETO) $(NETDO) $(ORDER_) $(MAINLIBS) $(NETLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libNet.$(SOEXT) $@ "$(NETO) $(NETDO)" \
		   "$(NETLIBEXTRA) $(CRYPTOLIBDIR) $(CRYPTOLIB) $(SSLLIB)"

$(call pcmrule,NET)
	$(noop)

$(NETDS):       $(NETINCH) $(NETL) $(ROOTCLINGEXE) $(call pcmdep,NET)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,NET) -c -writeEmptyRootPCM $(NETNOCRYPTO) $(NETSSL) $(patsubst include/%,%,$(NETINCH)) $(NETL)

$(NETMAP):      $(NETINCH) $(NETL) $(ROOTCLINGEXE) $(call pcmdep,NET)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(NETDS) $(call dictModule,NET) -c $(NETNOCRYPTO) $(NETSSL) $(patsubst include/%,%,$(NETINCH)) $(NETL)

all-$(MODNAME): $(NETLIB)

clean-$(MODNAME):
		@rm -f $(NETO) $(NETDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(NETDEP) $(NETDS) $(NETDH) $(NETLIB) $(NETMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(MACOSX_SSL_DEPRECATED),yes)
$(call stripsrc,$(NETDIRS)/TSSLSocket.o): CXXFLAGS += -Wno-deprecated-declarations
endif
$(call stripsrc,$(NETDIRS)/TSSLSocket.o): CXXFLAGS += $(SSLINCDIR:%=-I%)
$(call stripsrc,$(NETDIRS)/TS3HTTPRequest.o): CXXFLAGS += $(SSLINCDIR:%=-I%)
$(call stripsrc,$(NETDIRS)/TWebFile.o): CXXFLAGS += $(NETSSL)
