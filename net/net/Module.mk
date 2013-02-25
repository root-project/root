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

NETH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
NETS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
ifeq ($(CRYPTOLIB),)
NETNOCRYPTO  := -DR__NO_CRYPTO
NETH         := $(filter-out $(MODDIRI)/TS3WebFile.h,$(NETH))
NETH         := $(filter-out $(MODDIRI)/TS3HTTPRequest.h,$(NETH))
NETS         := $(filter-out $(MODDIRS)/TS3WebFile.cxx,$(NETS))
NETS         := $(filter-out $(MODDIRS)/TS3HTTPRequest.cxx,$(NETS))
else
NETNOCRYPTO  :=
endif

ifeq ($(SSLLIB),)
NETSSL       :=
NETH         := $(filter-out $(MODDIRI)/TSSLSocket.h,$(NETH))
NETS         := $(filter-out $(MODDIRS)/TSSLSocket.cxx,$(NETS))
else
NETSSL       := -DR__SSL
endif

NETO         := $(call stripsrc,$(NETS:.cxx=.o))
NETDEP       := $(NETO:.o=.d) $(NETDO:.o=.d)

NETLIB       := $(LPATH)/libNet.$(SOEXT)
NETMAP       := $(NETLIB:.$(SOEXT)=.rootmap)

EXTRANETFLAGS =

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(NETH))
ALLLIBS      += $(NETLIB)
ALLMAPS      += $(NETMAP)

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

$(NETDS):       $(NETH) $(NETL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(NETNOCRYPTO) $(NETSSL) $(NETH) $(NETL)

$(NETMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(NETL)
		$(RLIBMAP) -o $@ -l $(NETLIB) -d $(NETLIBDEPM) -c $(NETL)

all-$(MODNAME): $(NETLIB) $(NETMAP)

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
