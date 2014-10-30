# Module.mk for physics module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := physics
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PHYSICSDIR   := $(MODDIR)
PHYSICSDIRS  := $(PHYSICSDIR)/src
PHYSICSDIRI  := $(PHYSICSDIR)/inc

##### libPhysics #####
PHYSICSL     := $(MODDIRI)/LinkDef.h
PHYSICSDS    := $(call stripsrc,$(MODDIRS)/G__Physics.cxx)
PHYSICSDO    := $(PHYSICSDS:.cxx=.o)
PHYSICSDH    := $(PHYSICSDS:.cxx=.h)

PHYSICSH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PHYSICSS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PHYSICSO     := $(call stripsrc,$(PHYSICSS:.cxx=.o))

PHYSICSDEP   := $(PHYSICSO:.o=.d) $(PHYSICSDO:.o=.d)

PHYSICSLIB   := $(LPATH)/libPhysics.$(SOEXT)
PHYSICSMAP   := $(PHYSICSLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(PHYSICSH))
ALLLIBS     += $(PHYSICSLIB)
ALLMAPS     += $(PHYSICSMAP)

# include all dependency files
INCLUDEFILES += $(PHYSICSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(PHYSICSDIRI)/%.h
		cp $< $@

$(PHYSICSLIB):  $(PHYSICSO) $(PHYSICSDO) $(ORDER_) $(MAINLIBS) $(PHYSICSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPhysics.$(SOEXT) $@ \
		   "$(PHYSICSO) $(PHYSICSDO)" "$(PHYSICSLIBEXTRA)"

$(call pcmrule,PHYSICS)
	$(noop)

$(PHYSICSDS):   $(PHYSICSH) $(PHYSICSL) $(ROOTCLINGEXE) $(call pcmdep,PHYSICS)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,PHYSICS) -c -writeEmptyRootPCM $(PHYSICSH) $(PHYSICSL)

$(PHYSICSMAP):  $(PHYSICSH) $(PHYSICSL) $(ROOTCLINGEXE) $(call pcmdep,PHYSICS)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(PHYSICSDS) $(call dictModule,PHYSICS) -c $(PHYSICSH) $(PHYSICSL)

all-$(MODNAME): $(PHYSICSLIB)

clean-$(MODNAME):
		@rm -f $(PHYSICSO) $(PHYSICSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(PHYSICSDEP) $(PHYSICSDS) $(PHYSICSDH) $(PHYSICSLIB) $(PHYSICSMAP)

distclean::     distclean-$(MODNAME)
