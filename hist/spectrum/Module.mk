# Module.mk for spectrum module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 28/09/2006

MODNAME      := spectrum
MODDIR       := $(ROOT_SRCDIR)/hist/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SPECTRUMDIR  := $(MODDIR)
SPECTRUMDIRS := $(SPECTRUMDIR)/src
SPECTRUMDIRI := $(SPECTRUMDIR)/inc

##### libSpectrum #####
SPECTRUML    := $(MODDIRI)/LinkDef.h
SPECTRUMDS   := $(call stripsrc,$(MODDIRS)/G__Spectrum.cxx)
SPECTRUMDO   := $(SPECTRUMDS:.cxx=.o)
SPECTRUMDH   := $(SPECTRUMDS:.cxx=.h)

SPECTRUMH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SPECTRUMS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SPECTRUMO    := $(call stripsrc,$(SPECTRUMS:.cxx=.o))

SPECTRUMDEP  := $(SPECTRUMO:.o=.d) $(SPECTRUMDO:.o=.d)

SPECTRUMLIB  := $(LPATH)/libSpectrum.$(SOEXT)
SPECTRUMMAP  := $(SPECTRUMLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SPECTRUMH))
ALLLIBS      += $(SPECTRUMLIB)
ALLMAPS      += $(SPECTRUMMAP)

# include all dependency files
INCLUDEFILES += $(SPECTRUMDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SPECTRUMDIRI)/%.h
		cp $< $@

$(SPECTRUMLIB): $(SPECTRUMO) $(SPECTRUMDO) $(ORDER_) $(MAINLIBS) \
                $(SPECTRUMLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSpectrum.$(SOEXT) $@ \
		   "$(SPECTRUMO) $(SPECTRUMDO)" \
		   "$(SPECTRUMLIBEXTRA)"

$(call pcmrule,SPECTRUM)
	$(noop)

$(SPECTRUMDS):  $(SPECTRUMH) $(SPECTRUML) $(ROOTCLINGEXE) $(call pcmdep,SPECTRUM)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,SPECTRUM) -c -writeEmptyRootPCM $(SPECTRUMH) $(SPECTRUML)

$(SPECTRUMMAP): $(SPECTRUMH) $(SPECTRUML) $(ROOTCLINGEXE) $(call pcmdep,SPECTRUM)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(SPECTRUMDS) $(call dictModule,SPECTRUM) -c $(SPECTRUMH) $(SPECTRUML)

all-$(MODNAME): $(SPECTRUMLIB)

clean-$(MODNAME):
		@rm -f $(SPECTRUMO) $(SPECTRUMDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SPECTRUMDEP) $(SPECTRUMLIB) $(SPECTRUMMAP) \
		   $(SPECTRUMDS) $(SPECTRUMDH)

distclean::     distclean-$(MODNAME)
