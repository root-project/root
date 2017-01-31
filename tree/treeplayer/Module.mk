# Module.mk for treeplayer module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := treeplayer
MODDIR       := $(ROOT_SRCDIR)/tree/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TREEPLAYERDIR  := $(MODDIR)
TREEPLAYERDIRS := $(TREEPLAYERDIR)/src
TREEPLAYERDIRI := $(TREEPLAYERDIR)/inc

##### libTreePlayer #####
TREEPLAYERL  := $(MODDIRI)/LinkDef.h
TREEPLAYERDS := $(call stripsrc,$(MODDIRS)/G__TreePlayer.cxx)
TREEPLAYERDO := $(TREEPLAYERDS:.cxx=.o)
TREEPLAYERDH := $(TREEPLAYERDS:.cxx=.h)

TREEPLAYER2L  := $(MODDIRI)/DataFrameLinkDef.h
TREEPLAYER2DS := $(call stripsrc,$(MODDIRS)/G__DataFrame.cxx)
TREEPLAYER2DO := $(TREEPLAYER2DS:.cxx=.o)
TREEPLAYER2DH := $(TREEPLAYER2DS:.cxx=.h)

TREEPLAYERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TREEPLAYERH  := $(filter-out $(MODDIRI)/DataFrameLinkDef.h,$(TREEPLAYERH))
TREEPLAYERH  := $(filter-out $(MODDIRI)/TBranchProxyTemplate.h,$(TREEPLAYERH))
TREEPLAYERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TREEPLAYERS  := $(filter-out $(MODDIRS)/TTreeProcessor%.cxx,$(TREEPLAYERS))
TREEPLAYERO  := $(call stripsrc,$(TREEPLAYERS:.cxx=.o))

TREEPLAYER2H := $(wildcard $(MODDIRI)/ROOT/*.hxx)
TREEPLAYER2H := $(filter-out $(MODDIRI)/TTreeProcessorMT.h,$(TREEPLAYER2H))

TREEPLAYERDEP := $(TREEPLAYERO:.o=.d) $(TREEPLAYERDO:.o=.d)

TREEPLAYERLIB := $(LPATH)/libTreePlayer.$(SOEXT)
TREEPLAYERMAP := $(TREEPLAYERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
TREEPLAYERH_REL := $(patsubst $(MODDIRI)/%,include/%,$(TREEPLAYERH) $(TREEPLAYER2H))
ALLHDRS       += $(TREEPLAYERH_REL) $(MODDIRI)/TBranchProxyTemplate.h
ALLLIBS       += $(TREEPLAYERLIB)
ALLMAPS       += $(TREEPLAYERMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(TREEPLAYERH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Tree_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(TREEPLAYERLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(TREEPLAYERDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TREEPLAYERDIRI)/%.h
		cp $< $@

include/%.hxx:  $(TREEPLAYERDIRI)/%.hxx
		mkdir -p include/ROOT;
		cp $< $@

$(TREEPLAYERLIB): $(TREEPLAYERO) $(TREEPLAYERDO) $(TREEPLAYER2DO) $(ORDER_) $(MAINLIBS) \
                  $(TREEPLAYERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTreePlayer.$(SOEXT) $@ \
		   "$(TREEPLAYERO) $(TREEPLAYERDO) $(TREEPLAYER2DO)" \
		   "$(TREEPLAYERLIBEXTRA)"

$(call pcmrule,TREEPLAYER)
	$(noop)

$(TREEPLAYERDS): $(TREEPLAYERH) $(TREEPLAYERL) $(ROOTCLINGEXE) $(call pcmdep,TREEPLAYER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,TREEPLAYER) -c -writeEmptyRootPCM $(TREEPLAYERH) $(TREEPLAYERL)

$(TREEPLAYER2DS): $(TREEPLAYER2H) $(TREEPLAYER2L) $(ROOTCLINGEXE) $(call pcmdep,TREEPLAYER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,TREEPLAYER) -multiDict -writeEmptyRootPCM $(TREEPLAYER2H) $(TREEPLAYER2L)


$(TREEPLAYERMAP): $(TREEPLAYERH) $(TREEPLAYERL) $(ROOTCLINGEXE) $(call pcmdep,TREEPLAYER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(TREEPLAYERDS) $(call dictModule,TREEPLAYER) -c $(TREEPLAYERH) $(TREEPLAYERL)

all-$(MODNAME): $(TREEPLAYERLIB)

clean-$(MODNAME):
		@rm -f $(TREEPLAYERO) $(TREEPLAYERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TREEPLAYERDEP) $(TREEPLAYERDS) $(TREEPLAYERDH) \
		   $(TREEPLAYERLIB) $(TREEPLAYERMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(PLATFORM),macosx)
ifeq ($(GCC_VERS_FULL),gcc-4.0.1)
ifneq ($(filter -O%,$(OPT)),)
   $(call stripsrc,$(TREEPLAYERDIRS)/TTreeFormula.o): OPT = $(NOOPT)
endif
endif
ifeq ($(ICC_MAJOR),10)
   $(call stripsrc,$(TREEPLAYERDIRS)/TTreeFormula.o): OPT = $(NOOPT)
endif
endif

# Optimize dictionary with stl containers.
$(TREEPLAYERDO): NOOPT = $(OPT)
