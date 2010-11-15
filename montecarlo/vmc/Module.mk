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
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(VMCH))
ALLLIBS     += $(VMCLIB)
ALLMAPS     += $(VMCMAP)

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

$(VMCDS):       $(VMCH1) $(VMCL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(VMCH1) $(VMCL)

$(VMCMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(VMCL)
		$(RLIBMAP) -o $@ -l $(VMCLIB) \
		   -d $(VMCLIBDEPM) -c $(VMCL)

all-$(MODNAME): $(VMCLIB) $(VMCMAP)

clean-$(MODNAME):
		@rm -f $(VMCO) $(VMCDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(VMCDEP) $(VMCDS) $(VMCDH) $(VMCLIB) $(VMCMAP)

distclean::     distclean-$(MODNAME)
