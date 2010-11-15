# Module.mk for clarens module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Author: Maarten Ballintijn 18/10/2004

MODNAME      := clarens
MODDIR       := $(ROOT_SRCDIR)/proof/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CLARENSDIR   := $(MODDIR)
CLARENSDIRS  := $(CLARENSDIR)/src
CLARENSDIRI  := $(CLARENSDIR)/inc

##### libClarens #####
CLARENSL     := $(MODDIRI)/LinkDef.h
CLARENSDS    := $(call stripsrc,$(MODDIRS)/G__Clarens.cxx)
CLARENSDO    := $(CLARENSDS:.cxx=.o)
CLARENSDH    := $(CLARENSDS:.cxx=.h)

CLARENSH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CLARENSS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CLARENSO     := $(call stripsrc,$(CLARENSS:.cxx=.o))

CLARENSDEP   := $(CLARENSO:.o=.d) $(CLARENSDO:.o=.d)

CLARENSLIB   := $(LPATH)/libClarens.$(SOEXT)
CLARENSMAP   := $(CLARENSLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CLARENSH))
ALLLIBS     += $(CLARENSLIB)
ALLMAPS     += $(CLARENSMAP)

# include all dependency files
INCLUDEFILES += $(CLARENSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(CLARENSDIRI)/%.h
		cp $< $@

$(CLARENSLIB):  $(CLARENSO) $(CLARENSDO) $(ORDER_) $(MAINLIBS) $(CLARENSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libClarens.$(SOEXT) $@ \
		   "$(CLARENSO) $(CLARENSDO) $(CLARENSLIBS)" \
		   "$(CLARENSLIBEXTRA)"

$(CLARENSDS):   $(CLARENSH) $(CLARENSL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CLARENSH) $(CLARENSL)

$(CLARENSMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(CLARENSL)
		$(RLIBMAP) -o $@ -l $(CLARENSLIB) \
		   -d $(CLARENSLIBDEPM) -c $(CLARENSL)

all-$(MODNAME): $(CLARENSLIB) $(CLARENSMAP)

clean-$(MODNAME):
		@rm -f $(CLARENSO) $(CLARENSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(CLARENSDEP) $(CLARENSDS) $(CLARENSDH) $(CLARENSLIB) $(CLARENSMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(CLARENSO) $(CLARENSDO):    CXXFLAGS += $(CLARENSINC)
