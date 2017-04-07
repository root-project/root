# Module.mk for fumili module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 07/05/2003

MODNAME      := fumili
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FUMILIDIR    := $(MODDIR)
FUMILIDIRS   := $(FUMILIDIR)/src
FUMILIDIRI   := $(FUMILIDIR)/inc

##### libFumili #####
FUMILIL      := $(MODDIRI)/LinkDef.h
FUMILIDS     := $(call stripsrc,$(MODDIRS)/G__Fumili.cxx)
FUMILIDO     := $(FUMILIDS:.cxx=.o)
FUMILIDH     := $(FUMILIDS:.cxx=.h)

FUMILIH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FUMILIS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FUMILIO      := $(call stripsrc,$(FUMILIS:.cxx=.o))

FUMILIDEP    := $(FUMILIO:.o=.d) $(FUMILIDO:.o=.d)

FUMILILIB    := $(LPATH)/libFumili.$(SOEXT)
FUMILIMAP    := $(FUMILILIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
FUMILIH_REL := $(patsubst $(MODDIRI)/%.h,include/%.h,$(FUMILIH))
ALLHDRS     += $(FUMILIH_REL)
ALLLIBS     += $(FUMILILIB)
ALLMAPS     += $(FUMILIMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(FUMILIH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Math_Fumili { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(FUMILILIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(FUMILIDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FUMILIDIRI)/%.h
		cp $< $@

$(FUMILILIB):   $(FUMILIO) $(FUMILIDO) $(ORDER_) $(MAINLIBS) $(FUMILILIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFumili.$(SOEXT) $@ "$(FUMILIO) $(FUMILIDO)" \
		   "$(FUMILILIBEXTRA)"

$(call pcmrule,FUMILI)
	$(noop)

$(FUMILIDS):    $(FUMILIH) $(FUMILIL) $(ROOTCLINGEXE) $(call pcmdep,FUMILI)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,FUMILI) -c $(FUMILIH) $(FUMILIL)

$(FUMILIMAP):   $(FUMILIH) $(FUMILIL) $(ROOTCLINGEXE) $(call pcmdep,FUMILI)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(FUMILIDS) $(call dictModule,FUMILI) -c $(FUMILIH) $(FUMILIL)

all-$(MODNAME): $(FUMILILIB)

clean-$(MODNAME):
		@rm -f $(FUMILIO) $(FUMILIDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FUMILIDEP) $(FUMILIDS) $(FUMILIDH) $(FUMILILIB) $(FUMILIMAP)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(FUMILIDO): NOOPT = $(OPT)
