# Module.mk for table module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := table
MODDIR       := $(ROOT_SRCDIR)/misc/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TABLEDIR     := $(MODDIR)
TABLEDIRS    := $(TABLEDIR)/src
TABLEDIRI    := $(TABLEDIR)/inc

##### libTable #####
TABLEL       := $(MODDIRI)/LinkDef.h
TABLEDS      := $(call stripsrc,$(MODDIRS)/G__Table.cxx)
TABLEDO      := $(TABLEDS:.cxx=.o)
TABLEDH      := $(TABLEDS:.cxx=.h)

TABLEH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TABLES       := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TABLEO       := $(call stripsrc,$(TABLES:.cxx=.o))

TABLEDEP     := $(TABLEO:.o=.d) $(TABLEDO:.o=.d)

TABLELIB     := $(LPATH)/libTable.$(SOEXT)
TABLEMAP     := $(TABLELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
TABLEH_REL  := $(patsubst $(MODDIRI)/%.h,include/%.h,$(TABLEH))
ALLHDRS     += $(TABLEH_REL)
ALLLIBS     += $(TABLELIB)
ALLMAPS     += $(TABLEMAP)
ifeq ($(CXXMODULES),yes)
  CXXMODULES_HEADERS := $(patsubst include/%,header \"%\"\\n,$(TABLEH_REL))
  CXXMODULES_MODULEMAP_CONTENTS += module Misc_$(MODNAME) { \\n
  CXXMODULES_MODULEMAP_CONTENTS += $(CXXMODULES_HEADERS)
  CXXMODULES_MODULEMAP_CONTENTS += "export \* \\n"
  CXXMODULES_MODULEMAP_CONTENTS += link \"$(TABLELIB)\" \\n
  CXXMODULES_MODULEMAP_CONTENTS += } \\n
endif

# include all dependency files
INCLUDEFILES += $(TABLEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(TABLEDIRI)/%.h
		cp $< $@

$(TABLELIB):    $(TABLEO) $(TABLEDO) $(ORDER_) $(MAINLIBS) $(TABLELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTable.$(SOEXT) $@ "$(TABLEO) $(TABLEDO)" \
		   "$(TABLELIBEXTRA)"

$(call pcmrule,TABLE)
	$(noop)

$(TABLEDS):     $(TABLEH) $(TABLEL) $(ROOTCLINGEXE) $(call pcmdep,TABLE)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,TABLE) -c $(TABLEH) $(TABLEL)

$(TABLEMAP):    $(TABLEH) $(TABLEL) $(ROOTCLINGEXE) $(call pcmdep,TABLE)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(TABLEDS) $(call dictModule,TABLE) -c $(TABLEH) $(TABLEL)

all-$(MODNAME): $(TABLELIB)

clean-$(MODNAME):
		@rm -f $(TABLEO) $(TABLEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TABLEDEP) $(TABLEDS) $(TABLEDH) $(TABLELIB) $(TABLEMAP)

distclean::     distclean-$(MODNAME)
