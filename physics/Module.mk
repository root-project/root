# Module.mk for physics module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := physics
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

PHYSICSDIR   := $(MODDIR)
PHYSICSDIRS  := $(PHYSICSDIR)/src
PHYSICSDIRI  := $(PHYSICSDIR)/inc

##### libPhysics #####
PHYSICSL     := $(MODDIRI)/LinkDef.h
PHYSICSDS    := $(MODDIRS)/G__Physics.cxx
PHYSICSDO    := $(PHYSICSDS:.cxx=.o)
PHYSICSDH    := $(PHYSICSDS:.cxx=.h)

PHYSICSH     := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
PHYSICSS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
PHYSICSO     := $(PHYSICSS:.cxx=.o)

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
include/%.h:    $(PHYSICSDIRI)/%.h
		cp $< $@

$(PHYSICSLIB):  $(PHYSICSO) $(PHYSICSDO) $(ORDER_) $(MAINLIBS) $(PHYSICSLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libPhysics.$(SOEXT) $@ \
		   "$(PHYSICSO) $(PHYSICSDO)" "$(PHYSICSLIBEXTRA)"

$(PHYSICSDS):   $(PHYSICSH) $(PHYSICSL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(PHYSICSH) $(PHYSICSL)

$(PHYSICSMAP):  $(RLIBMAP) $(MAKEFILEDEP) $(PHYSICSL)
		$(RLIBMAP) -o $(PHYSICSMAP) -l $(PHYSICSLIB) \
		   -d $(PHYSICSLIBDEPM) -c $(PHYSICSL)

all-physics:    $(PHYSICSLIB) $(PHYSICSMAP)

clean-physics:
		@rm -f $(PHYSICSO) $(PHYSICSDO)

clean::         clean-physics

distclean-physics: clean-physics
		@rm -f $(PHYSICSDEP) $(PHYSICSDS) $(PHYSICSDH) $(PHYSICSLIB) $(PHYSICSMAP)

distclean::     distclean-physics
