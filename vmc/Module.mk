# Module.mk for vmc module
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 12/4/2002

MODDIR       := vmc
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

VMCDIR       := $(MODDIR)
VMCDIRS      := $(VMCDIR)/src
VMCDIRI      := $(VMCDIR)/inc

##### libVMC #####
VMCL         := $(MODDIRI)/LinkDef.h
VMCDS        := $(MODDIRS)/G__VMC.cxx
VMCDO        := $(VMCDS:.cxx=.o)
VMCDH        := $(VMCDS:.cxx=.h)

VMCH1        := $(wildcard $(MODDIRI)/T*.h)
VMCH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
VMCS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
VMCO         := $(VMCS:.cxx=.o)

VMCDEP       := $(VMCO:.o=.d) $(VMCDO:.o=.d)

VMCLIB       := $(LPATH)/libVMC.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(VMCH))
ALLLIBS     += $(VMCLIB)

# include all dependency files
INCLUDEFILES += $(VMCDEP)

##### local rules #####
include/%.h:    $(VMCDIRI)/%.h
		cp $< $@

$(VMCLIB):      $(VMCO) $(VMCDO) $(MAINLIBS) $(VMCLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libVMC.$(SOEXT) $@ "$(VMCO) $(VMCDO)" \
		   "$(VMCLIBEXTRA)"

$(VMCDS):       $(VMCH1) $(VMCL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(VMCH1) $(VMCL)

$(VMCDO):       $(VMCDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-vmc:        $(VMCLIB)

map-vmc:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(VMCLIB) \
		   -d $(VMCLIBDEP) -c $(VMCL)

map::           map-vmc

clean-vmc:
		@rm -f $(VMCO) $(VMCDO)

clean::         clean-vmc

distclean-vmc:   clean-vmc
		@rm -f $(VMCDEP) $(VMCDS) $(VMCDH) $(VMCLIB)

distclean::     distclean-vmc
