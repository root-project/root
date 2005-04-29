# Module.mk for proof module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := proof
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PROOFDIR     := $(MODDIR)
PROOFDIRS    := $(PROOFDIR)/src
PROOFDIRI    := $(PROOFDIR)/inc

##### libProof #####
PROOFL       := $(MODDIRI)/LinkDef.h
PROOFDS      := $(MODDIRS)/G__Proof.cxx
PROOFDO      := $(PROOFDS:.cxx=.o)
PROOFDH      := $(PROOFDS:.cxx=.h)

PROOFH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PROOFS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PROOFO       := $(PROOFS:.cxx=.o)

PROOFDEP     := $(PROOFO:.o=.d) $(PROOFDO:.o=.d)

PROOFLIB     := $(LPATH)/libProof.$(SOEXT)

##### libProofGui #####
PROOFGUIL    := $(MODDIRI)/LinkDefGui.h
PROOFGUIDS   := $(MODDIRS)/G__ProofGui.cxx
PROOFGUIDO   := $(PROOFGUIDS:.cxx=.o)
PROOFGUIDH   := $(PROOFGUIDS:.cxx=.h)

PROOFGUIH    := $(MODDIRI)/TProofProgressDialog.h
PROOFGUIS    := $(MODDIRS)/TProofProgressDialog.cxx
PROOFGUIO    := $(PROOFGUIS:.cxx=.o)

PROOFGUIDEP  := $(PROOFGUIO:.o=.d) $(PROOFGUIDO:.o=.d)

PROOFGUILIB  := $(LPATH)/libProofGui.$(SOEXT)

# remove GUI files from PROOF files
PROOFH       := $(filter-out $(PROOFGUIH),$(PROOFH))
PROOFS       := $(filter-out $(PROOFGUIS),$(PROOFS))
PROOFO       := $(filter-out $(PROOFGUIO),$(PROOFO))
PROOFDEP     := $(filter-out $(PROOFGUIDEP),$(PROOFDEP))

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFH))
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PROOFGUIH))
ALLLIBS     += $(PROOFLIB) $(PROOFGUILIB)

# include all dependency files
INCLUDEFILES += $(PROOFDEP) $(PROOFGUIDEP)

##### local rules #####
include/%.h:    $(PROOFDIRI)/%.h
		cp $< $@

$(PROOFLIB):    $(PROOFO) $(PROOFDO) $(MAINLIBS) $(PROOFLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProof.$(SOEXT) $@ "$(PROOFO) $(PROOFDO)" \
		   "$(PROOFLIBEXTRA)"

$(PROOFDS):     $(PROOFH) $(PROOFL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PROOFH) $(PROOFL)

$(PROOFDO):     $(PROOFDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

$(PROOFGUILIB): $(PROOFGUIO) $(PROOFGUIDO) $(MAINLIBS) $(PROOFGUILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libProofGui.$(SOEXT) $@ \
		   "$(PROOFGUIO) $(PROOFGUIDO)" \
		   "$(PROOFGUILIBEXTRA)"

$(PROOFGUIDS):  $(PROOFGUIH) $(PROOFGUIL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PROOFGUIH) $(PROOFGUIL)

$(PROOFGUIDO):  $(PROOFGUIDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-proof:      $(PROOFLIB) $(PROOFGUILIB)

map-proof:      $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PROOFLIB) \
		   -d $(PROOFLIBDEP) -c $(PROOFL)

map-proofgui:   $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(PROOFGUILIB) \
		   -d $(PROOFGUILIBDEP) -c $(PROOFGUIL)

map::           map-proof map-proofgui

clean-proof:
		@rm -f $(PROOFO) $(PROOFDO) $(PROOFGUIO) $(PROOFGUIDO)

clean::         clean-proof

distclean-proof: clean-proof
		@rm -f $(PROOFDEP) $(PROOFDS) $(PROOFDH) $(PROOFLIB) \
		   $(PROOFGUIDEP) $(PROOFGUIDS) $(PROOFGUIDH) \
		   $(PROOFGUILIB)

distclean::     distclean-proof
