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

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(CLARENSH))
ALLLIBS     += $(CLARENSLIB)

# include all dependency files
INCLUDEFILES += $(CLARENSDEP)

##### local rules #####
include/%.h:    $(CLARENSDIRI)/%.h
		cp $< $@

$(CLARENSLIB):  $(CLARENSO) $(CLARENSDO) $(MAINLIBS) $(CLARENSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libClarens.$(SOEXT) $@ \
		   "$(CLARENSO) $(CLARENSDO) $(CLARENSLIBS)" \
		   "$(CLARENSLIBEXTRA)"

$(CLARENSDS):   $(CLARENSH) $(CLARENSL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(CLARENSH) $(CLARENSL)

$(CLARENSDO):   $(CLARENSDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) $(CLARENSINC) -I. -o $@ -c $<

all-clarens:    $(CLARENSLIB)

map-clarens:    $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(CLARENSLIB) \
		   -d $(CLARENSLIBDEP) -c $(CLARENSL)

map::           map-clarens

clean-clarens:
		@rm -f $(CLARENSO) $(CLARENSDO)

clean::         clean-clarens

distclean-clarens: clean-clarens
		@rm -f $(CLARENSDEP) $(CLARENSDS) $(CLARENSDH) $(CLARENSLIB)

distclean::     distclean-clarens

##### extra rules ######
$(CLARENSO):    %.o: %.cxx
		$(CXX) $(OPT) $(CXXFLAGS) $(CLARENSINC) -o $@ -c $<
