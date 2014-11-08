# Module.mk for gpad module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := gpad
MODDIR       := $(ROOT_SRCDIR)/graf2d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GPADDIR      := $(MODDIR)
GPADDIRS     := $(GPADDIR)/src
GPADDIRI     := $(GPADDIR)/inc

##### libGpad #####
GPADL        := $(MODDIRI)/LinkDef.h
GPADDS       := $(call stripsrc,$(MODDIRS)/G__Gpad.cxx)
GPADDO       := $(GPADDS:.cxx=.o)
GPADDH       := $(GPADDS:.cxx=.h)

GPADH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GPADS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GPADO        := $(call stripsrc,$(GPADS:.cxx=.o))

GPADDEP      := $(GPADO:.o=.d) $(GPADDO:.o=.d)

GPADLIB      := $(LPATH)/libGpad.$(SOEXT)
GPADMAP      := $(GPADLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GPADH))
ALLLIBS     += $(GPADLIB)
ALLMAPS     += $(GPADMAP)

# include all dependency files
INCLUDEFILES += $(GPADDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GPADDIRI)/%.h
		cp $< $@

$(GPADLIB):     $(GPADO) $(GPADDO) $(ORDER_) $(MAINLIBS) $(GPADLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGpad.$(SOEXT) $@ "$(GPADO) $(GPADDO)" \
		   "$(GPADLIBEXTRA)"

$(call pcmrule,GPAD)
	$(noop)

$(GPADDS):      $(GPADH) $(GPADL) $(ROOTCLINGEXE) $(call pcmdep,GPAD)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GPAD) -c -writeEmptyRootPCM $(GPADH) $(GPADL)

$(GPADMAP):     $(GPADH) $(GPADL) $(ROOTCLINGEXE) $(call pcmdep,GPAD)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GPADDS) $(call dictModule,GPAD) -c $(GPADH) $(GPADL)

all-$(MODNAME): $(GPADLIB)

clean-$(MODNAME):
		@rm -f $(GPADO) $(GPADDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GPADDEP) $(GPADDS) $(GPADDH) $(GPADLIB) $(GPADMAP)

distclean::     distclean-$(MODNAME)
