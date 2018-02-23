# Module.mk for http module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/11/2002

MODNAME      := httpsniff
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HTTPSNIFFDIR      := $(MODDIR)
HTTPSNIFFDIRS     := $(HTTPSNIFFDIR)/src
HTTPSNIFFDIRI     := $(HTTPSNIFFDIR)/inc

##### libRHTTPSniff #####
HTTPSNIFFL        := $(MODDIRI)/LinkDef.h
HTTPSNIFFDS       := $(call stripsrc,$(MODDIRS)/G__RHTTPSniff.cxx)
HTTPSNIFFDO       := $(HTTPSNIFFDS:.cxx=.o)
HTTPSNIFFDH       := $(HTTPSNIFFDS:.cxx=.h)

HTTPSNIFFH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HTTPSNIFFS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HTTPSNIFFO        := $(call stripsrc,$(HTTPSNIFFS:.cxx=.o))

HTTPSNIFFDEP      := $(HTTPSNIFFO:.o=.d) $(HTTPSNIFFDO:.o=.d)

HTTPSNIFFLIB      := $(LPATH)/libRHTTPSniff.$(SOEXT)
HTTPSNIFFMAP      := $(HTTPSNIFFLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
HTTPSNIFFH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(HTTPSNIFFH))
ALLHDRS     += $(HTTPSNIFFH_REL)
ALLLIBS     += $(HTTPSNIFFLIB)
ALLMAPS     += $(HTTPSNIFFMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(HTTPSNIFFH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Net_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(HTTPSNIFFLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(HTTPSNIFFDEP)


##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(HTTPSNIFFDIRI)/%.h
		cp $< $@

$(HTTPSNIFFLIB):   $(HTTPSNIFFO) $(HTTPSNIFFDO) $(ORDER_) $(MAINLIBS) $(HTTPSNIFFLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libRHTTPSniff.$(SOEXT) $@ "$(HTTPSNIFFO) $(HTTPSNIFFDO)" \
		   "$(HTTPSNIFFLIBEXTRA) $(HTTPSNIFFLIBDIR)"

# this is raplacement for #$(call pcmrule,RHTTP)
lib/libRHTTPSniff_rdict.pcm: lib/libCore_rdict.pcm  net/httpsniff/src/G__RHTTPSniff.cxx
	$(noop)

# this is raplacement for $(call dictModule,RHTTP)
DICTMODULE_RHHTP = -s lib/libRHTTPSniff.$(SOEXT) -rml libRHTTPSniff.$(SOEXT) -rmf lib/libRHTTPSniff.rootmap -m lib/libCore_rdict.pcm

$(HTTPSNIFFDS): $(HTTPSNIFFH) $(HTTPSNIFFL) $(ROOTCLINGEXE) $(call pcmdep,RHTTPSniff)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(DICTMODULE_RHHTP) -c $(HTTPSNIFFH) $(HTTPSNIFFL)

$(HTTPSNIFFMAP):     $(HTTPSNIFFH) $(HTTPSNIFFL) $(ROOTCLINGEXE) $(call pcmdep,RHTTPSniff)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(HTTPSNIFFDS) $(DICTMODULE_RHHTP) -c $(HTTPSNIFFH) $(HTTPSNIFFL)

all-$(MODNAME): $(HTTPSNIFFLIB)

clean-$(MODNAME):
		@rm -f $(HTTPSNIFFO) $(HTTPSNIFFDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HTTPSNIFFDEP) $(HTTPSNIFFDS) $(HTTPSNIFFDH) $(HTTPSNIFFLIB) $(HTTPSNIFFMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(HTTPSNIFFO) $(HTTPSNIFFDO) : CXXFLAGS += $(HTTPCXXFLAGS)
