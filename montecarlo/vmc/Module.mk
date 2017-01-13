# Module.mk for vmc module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/4/2002

MODNAME      := vmc
MODDIR       := $(ROOT_SRCDIR)/montecarlo/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

VMCDIR       := $(MODDIR)
VMCDIRS      := $(VMCDIR)/src
VMCDIRI      := $(VMCDIR)/inc

##### libVMC #####
VMCL         := $(MODDIRI)/LinkDef.h
VMCDS        := $(call stripsrc,$(MODDIRS)/G__VMC.cxx)
VMCDO        := $(VMCDS:.cxx=.o)
VMCDH        := $(VMCDS:.cxx=.h)

VMCH1        := $(wildcard $(MODDIRI)/T*.h)
VMCH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
VMCS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
VMCO         := $(call stripsrc,$(VMCS:.cxx=.o))

VMCDEP       := $(VMCO:.o=.d) $(VMCDO:.o=.d)

VMCLIB       := $(LPATH)/libVMC.$(SOEXT)
VMCMAP       := $(VMCLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
VMCH_REL    := $(patsubst $(MODDIRI)/%.h,include/%.h,$(VMCH))
ALLHDRS     += $(VMCH_REL)
ALLLIBS     += $(VMCLIB)
ALLMAPS     += $(VMCMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(VMCH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Montecarlo_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(VMCLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(VMCDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(VMCDIRI)/%.h
		cp $< $@

$(VMCLIB):      $(VMCO) $(VMCDO) $(ORDER_) $(MAINLIBS) $(VMCLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libVMC.$(SOEXT) $@ "$(VMCO) $(VMCDO)" \
		   "$(VMCLIBEXTRA)"

$(call pcmrule,VMC)
	$(noop)

$(VMCDS):       $(VMCH1) $(VMCL) $(ROOTCLINGEXE) $(call pcmdep,VMC)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,VMC) -c $(VMCH1) $(VMCL)

$(VMCMAP):      $(VMCH1) $(VMCL) $(ROOTCLINGEXE) $(call pcmdep,VMC)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(VMCDS) $(call dictModule,VMC) -c $(VMCH1) $(VMCL)

all-$(MODNAME): $(VMCLIB)

clean-$(MODNAME):
		@rm -f $(VMCO) $(VMCDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(VMCDEP) $(VMCDS) $(VMCDH) $(VMCLIB) $(VMCMAP)

distclean::     distclean-$(MODNAME)
