# Module.mk for clarens module
# Copyright (c) 2004 Rene Brun and Fons Rademakers
#
# Author: Maarten Ballintijn 18/10/2004

MODDIR       := clarens
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

CLARENSDIR   := $(MODDIR)
CLARENSDIRS  := $(CLARENSDIR)/src
CLARENSDIRI  := $(CLARENSDIR)/inc

##### libClarens #####
CLARENSL     := $(MODDIRI)/LinkDef.h
CLARENSDS    := $(MODDIRS)/G__Clarens.cxx
CLARENSDO    := $(CLARENSDS:.cxx=.o)
CLARENSDH    := $(CLARENSDS:.cxx=.h)

CLARENSH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
CLARENSS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
CLARENSO     := $(CLARENSS:.cxx=.o)

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
include/%.h:    $(CLARENSDIRI)/%.h
		cp $< $@

$(CLARENSLIB):  $(CLARENSO) $(CLARENSDO) $(ORDER_) $(MAINLIBS) $(CLARENSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libClarens.$(SOEXT) $@ \
		   "$(CLARENSO) $(CLARENSDO) $(CLARENSLIBS)" \
		   "$(CLARENSLIBEXTRA)"

$(CLARENSDS):   $(CLARENSH) $(CLARENSL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CLARENSH) $(CLARENSL)

$(CLARENSMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(CLARENSL)
		$(RLIBMAP) -o $(CLARENSMAP) -l $(CLARENSLIB) \
		   -d $(CLARENSLIBDEPM) -c $(CLARENSL)

all-clarens:    $(CLARENSLIB) $(CLARENSMAP)

clean-clarens:
		@rm -f $(CLARENSO) $(CLARENSDO)

clean::         clean-clarens

distclean-clarens: clean-clarens
		@rm -f $(CLARENSDEP) $(CLARENSDS) $(CLARENSDH) $(CLARENSLIB) $(CLARENSMAP)

distclean::     distclean-clarens

##### extra rules ######
$(CLARENSO) $(CLARENSDO):    CXXFLAGS += $(CLARENSINC)
