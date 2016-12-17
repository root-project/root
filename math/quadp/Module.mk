# Module.mk for quadp module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, Eddy Offermann, 21/05/2003

MODNAME      := quadp
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

QUADPDIR     := $(MODDIR)
QUADPDIRS    := $(QUADPDIR)/src
QUADPDIRI    := $(QUADPDIR)/inc

##### libQuadp #####
QUADPL       := $(MODDIRI)/LinkDef.h
QUADPDS      := $(call stripsrc,$(MODDIRS)/G__Quadp.cxx)
QUADPDO      := $(QUADPDS:.cxx=.o)
QUADPDH      := $(QUADPDS:.cxx=.h)

QUADPH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
QUADPS       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
QUADPO       := $(call stripsrc,$(QUADPS:.cxx=.o))

QUADPDEP     := $(QUADPO:.o=.d) $(QUADPDO:.o=.d)

QUADPLIB     := $(LPATH)/libQuadp.$(SOEXT)
QUADPMAP     := $(QUADPLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
QUADPH_REL  := $(patsubst $(MODDIRI)/%.h,include/%.h,$(QUADPH))
ALLHDRS     += $(QUADPH_REL)
ALLLIBS     += $(QUADPLIB)
ALLMAPS     += $(QUADPMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(QUADPH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Math_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(QUADPLIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(QUADPDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(QUADPDIRI)/%.h
		cp $< $@

$(QUADPLIB):    $(QUADPO) $(QUADPDO) $(ORDER_) $(MAINLIBS) $(QUADPLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libQuadp.$(SOEXT) $@ "$(QUADPO) $(QUADPDO)" \
		   "$(QUADPLIBEXTRA)"

$(call pcmrule,QUADP)
	$(noop)

$(QUADPDS):     $(QUADPH) $(QUADPL) $(ROOTCLINGEXE) $(call pcmdep,QUADP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,QUADP) -c $(QUADPH) $(QUADPL)

$(QUADPMAP):    $(QUADPH) $(QUADPL) $(ROOTCLINGEXE) $(call pcmdep,QUADP)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(QUADPDS) $(call dictModule,QUADP) -c $(QUADPH) $(QUADPL)

all-$(MODNAME): $(QUADPLIB)

clean-$(MODNAME):
		@rm -f $(QUADPO) $(QUADPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(QUADPDEP) $(QUADPDS) $(QUADPDH) $(QUADPLIB) $(QUADPMAP)

distclean::     distclean-$(MODNAME)
