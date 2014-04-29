# Module.mk for ged module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Ilka Antcheva, 18/2/2004

MODNAME   := ged
MODDIR    := $(ROOT_SRCDIR)/gui/$(MODNAME)
MODDIRS   := $(MODDIR)/src
MODDIRI   := $(MODDIR)/inc

GEDDIR    := $(MODDIR)
GEDDIRS   := $(GEDDIR)/src
GEDDIRI   := $(GEDDIR)/inc

##### libGed #####
GEDL      := $(MODDIRI)/LinkDef.h
GEDDS     := $(call stripsrc,$(MODDIRS)/G__Ged.cxx)
GEDDO     := $(GEDDS:.cxx=.o)
GEDDH     := $(GEDDS:.cxx=.h)

GEDH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GEDS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEDO      := $(call stripsrc,$(GEDS:.cxx=.o))

GEDDEP    := $(GEDO:.o=.d) $(GEDDO:.o=.d)

GEDLIB    := $(LPATH)/libGed.$(SOEXT)
GEDMAP    := $(GEDLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEDH))
ALLLIBS     += $(GEDLIB)
ALLMAPS     += $(GEDMAP)

# include all dependency files
INCLUDEFILES += $(GEDDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GEDDIRI)/%.h
		cp $< $@

$(GEDLIB):      $(GEDO) $(GEDDO) $(ORDER_) $(MAINLIBS) $(GEDLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGed.$(SOEXT) $@ "$(GEDO) $(GEDDO)" \
		   "$(GEDLIBEXTRA)"

$(call pcmrule,GED)
	$(noop)

$(GEDDS):       $(GEDH) $(GEDL) $(ROOTCLINGEXE) $(call pcmdep,GED)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GED) -c $(GEDH) $(GEDL)

$(GEDMAP):      $(GEDH) $(GEDL) $(ROOTCLINGEXE) $(call pcmdep,GED)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GEDDS) $(call dictModule,GED) -c $(GEDH) $(GEDL)

all-$(MODNAME): $(GEDLIB)

clean-$(MODNAME):
		@rm -f $(GEDO) $(GEDDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GEDDEP) $(GEDDS) $(GEDDH) $(GEDLIB) $(GEDMAP)

distclean::     distclean-$(MODNAME)
