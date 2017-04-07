# Module.mk for http module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/11/2002

MODNAME      := http
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HTTPDIR      := $(MODDIR)
HTTPDIRS     := $(HTTPDIR)/src
HTTPDIRI     := $(HTTPDIR)/inc
CIVETWEBDIR  := $(HTTPDIR)/civetweb

HTTPCLILIB   := $(OSTHREADLIB)
HTTPINCDIR   := $(CIVETWEBDIR) 

##### libRHTTP #####
HTTPL        := $(MODDIRI)/LinkDef.h
HTTPDS       := $(call stripsrc,$(MODDIRS)/G__RHTTP.cxx)
HTTPDO       := $(HTTPDS:.cxx=.o)
HTTPDH       := $(HTTPDS:.cxx=.h)

HTTPH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HTTPS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HTTPO        := $(call stripsrc,$(HTTPS:.cxx=.o))

CIVETWEBS    := $(CIVETWEBDIR)/civetweb.c
CIVETWEBO    := $(call stripsrc,$(CIVETWEBS:.c=.o))

HTTPDEP      := $(HTTPO:.o=.d) $(HTTPDO:.o=.d)

HTTPLIB      := $(LPATH)/libRHTTP.$(SOEXT)
HTTPMAP      := $(HTTPLIB:.$(SOEXT)=.rootmap)

HTTPCXXFLAGS := $(HTTPINCDIR:%=-I%) $(FASTCGIINCDIR:%=-I%) $(FASTCGIFLAGS) -DUSE_WEBSOCKET

# used in the main Makefile
HTTPH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(HTTPH))
ALLHDRS     += $(HTTPH_REL)
ALLLIBS     += $(HTTPLIB)
ALLMAPS     += $(HTTPMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(HTTPH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Net_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(HTTPLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(HTTPDEP)

HTTPLIBEXTRA += $(ZLIBLIBDIR) $(ZLIBCLILIB)

ifeq ($(PLATFORM),linux)
HTTPLIBEXTRA += -lrt
endif


##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(HTTPDIRI)/%.h
		cp $< $@

$(HTTPLIB):     $(HTTPO) $(HTTPDO) $(CIVETWEBO) $(ORDER_) $(MAINLIBS) $(HTTPLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRHTTP.$(SOEXT) $@ "$(HTTPO) $(HTTPDO) $(CIVETWEBO)" \
		   "$(HTTPLIBEXTRA) $(HTTPLIBDIR) $(HTTPCLILIB)"

# this is raplacement for #$(call pcmrule,RHTTP)
lib/libRHTTP_rdict.pcm: lib/libCore_rdict.pcm  net/http/src/G__RHTTP.cxx
	$(noop)

# this is raplacement for $(call dictModule,RHTTP)
DICTMODULE_RHHTP = -s lib/libRHTTP.$(SOEXT) -rml libRHTTP.$(SOEXT) -rmf lib/libRHTTP.rootmap -m lib/libCore_rdict.pcm

$(HTTPDS):      $(HTTPH) $(HTTPL) $(ROOTCLINGEXE) $(call pcmdep,RHTTP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(DICTMODULE_RHHTP) -c $(HTTPH) $(HTTPL)

$(HTTPMAP):     $(HTTPH) $(HTTPL) $(ROOTCLINGEXE) $(call pcmdep,RHTTP)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(HTTPDS) $(DICTMODULE_RHHTP) -c $(HTTPH) $(HTTPL)

#$(call pcmrule,RHTTP)
#	$(noop)

#$(HTTPDS):      $(HTTPH) $(HTTPL) $(ROOTCLINGEXE) $(call pcmdep,RHTTP)
#		$(MAKEDIR)
#		@echo "Generating dictionary $@..."
#		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,RHTTP) -c -I$(ROOT_SRCDIR) $(HTTPH) $(HTTPL)

#$(HTTPMAP):     $(HTTPH) $(HTTPL) $(ROOTCLINGEXE) $(call pcmdep,RHTTP)
#		$(MAKEDIR)
#		@echo "Generating rootmap $@..."
#		$(ROOTCLINGSTAGE2) -r $(HTTPDS) $(call dictModule,RHHTP) -c -I$(ROOT_SRCDIR) $(HTTPH) $(HTTPL)

all-$(MODNAME): $(HTTPLIB)

clean-$(MODNAME):
		@rm -f $(HTTPO) $(HTTPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HTTPDEP) $(HTTPDS) $(HTTPDH) $(HTTPLIB) $(HTTPMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(HTTPO) $(HTTPDO) : CXXFLAGS += $(HTTPCXXFLAGS)

$(CIVETWEBO) : CFLAGS += -DUSE_WEBSOCKET
