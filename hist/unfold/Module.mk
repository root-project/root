# Module.mk for unfold module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Olivier Couet, 23/11/2016

MODNAME      := unfold
MODDIR       := $(ROOT_SRCDIR)/hist/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UNFOLDDIR  := $(MODDIR)
UNFOLDDIRS := $(UNFOLDDIR)/src
UNFOLDDIRI := $(UNFOLDDIR)/inc

##### libSpectrum #####
UNFOLDL    := $(MODDIRI)/LinkDef.h
UNFOLDDS   := $(call stripsrc,$(MODDIRS)/G__Spectrum.cxx)
UNFOLDDO   := $(UNFOLDDS:.cxx=.o)
UNFOLDDH   := $(UNFOLDDS:.cxx=.h)

UNFOLDH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
UNFOLDS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
UNFOLDO    := $(call stripsrc,$(UNFOLDS:.cxx=.o))

UNFOLDDEP  := $(UNFOLDO:.o=.d) $(UNFOLDDO:.o=.d)

UNFOLDLIB  := $(LPATH)/libSpectrum.$(SOEXT)
UNFOLDMAP  := $(UNFOLDLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(UNFOLDH))
ALLLIBS      += $(UNFOLDLIB)
ALLMAPS      += $(UNFOLDMAP)

# include all dependency files
INCLUDEFILES += $(UNFOLDDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(UNFOLDDIRI)/%.h
		cp $< $@

$(UNFOLDLIB): $(UNFOLDO) $(UNFOLDDO) $(ORDER_) $(MAINLIBS) \
                $(UNFOLDLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSpectrum.$(SOEXT) $@ \
		   "$(UNFOLDO) $(UNFOLDDO)" \
		   "$(UNFOLDLIBEXTRA)"

$(call pcmrule,UNFOLD)
	$(noop)

$(UNFOLDDS):  $(UNFOLDH) $(UNFOLDL) $(ROOTCLINGEXE) $(call pcmdep,UNFOLD)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,UNFOLD) -c -writeEmptyRootPCM $(UNFOLDH) $(UNFOLDL)

$(UNFOLDMAP): $(UNFOLDH) $(UNFOLDL) $(ROOTCLINGEXE) $(call pcmdep,UNFOLD)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(UNFOLDDS) $(call dictModule,UNFOLD) -c $(UNFOLDH) $(UNFOLDL)

all-$(MODNAME): $(UNFOLDLIB)

clean-$(MODNAME):
		@rm -f $(UNFOLDO) $(UNFOLDDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(UNFOLDDEP) $(UNFOLDLIB) $(UNFOLDMAP) \
		   $(UNFOLDDS) $(UNFOLDDH)

distclean::     distclean-$(MODNAME)
