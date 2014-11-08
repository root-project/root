# Module.mk for spectrumpainter module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Olivier Couet, 27/11/2006

MODNAME      := spectrumpainter
MODDIR       := $(ROOT_SRCDIR)/hist/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SPECTRUMPAINTERDIR  := $(MODDIR)
SPECTRUMPAINTERDIRS := $(SPECTRUMPAINTERDIR)/src
SPECTRUMPAINTERDIRI := $(SPECTRUMPAINTERDIR)/inc

##### libSpectrumPainter #####
SPECTRUMPAINTERL  := $(MODDIRI)/LinkDef.h
SPECTRUMPAINTERDS := $(call stripsrc,$(MODDIRS)/G__SpectrumPainter.cxx)
SPECTRUMPAINTERDO := $(SPECTRUMPAINTERDS:.cxx=.o)
SPECTRUMPAINTERDH := $(SPECTRUMPAINTERDS:.cxx=.h)

SPECTRUMPAINTERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SPECTRUMPAINTERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SPECTRUMPAINTERO  := $(call stripsrc,$(SPECTRUMPAINTERS:.cxx=.o))

SPECTRUMPAINTERDEP := $(SPECTRUMPAINTERO:.o=.d) $(SPECTRUMPAINTERDO:.o=.d)

SPECTRUMPAINTERLIB := $(LPATH)/libSpectrumPainter.$(SOEXT)
SPECTRUMPAINTERMAP := $(SPECTRUMPAINTERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SPECTRUMPAINTERH))
ALLLIBS       += $(SPECTRUMPAINTERLIB)
ALLMAPS       += $(SPECTRUMPAINTERMAP)

# include all dependency files
INCLUDEFILES += $(SPECTRUMPAINTERDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SPECTRUMPAINTERDIRI)/%.h
		cp $< $@

$(SPECTRUMPAINTERLIB): $(SPECTRUMPAINTERO) $(SPECTRUMPAINTERDO) $(ORDER_) \
                       $(MAINLIBS) $(SPECTRUMPAINTERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSpectrumPainter.$(SOEXT) $@ \
		   "$(SPECTRUMPAINTERO) $(SPECTRUMPAINTERDO)" \
		   "$(SPECTRUMPAINTERLIBEXTRA)"

$(call pcmrule,SPECTRUMPAINTER)
	$(noop)

$(SPECTRUMPAINTERDS): $(SPECTRUMPAINTERH) $(SPECTRUMPAINTERL) $(ROOTCLINGEXE) $(call pcmdep,SPECTRUMPAINTER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,SPECTRUMPAINTER) -c -writeEmptyRootPCM $(SPECTRUMPAINTERH) $(SPECTRUMPAINTERL)

$(SPECTRUMPAINTERMAP): $(SPECTRUMPAINTERH) $(SPECTRUMPAINTERL) $(ROOTCLINGEXE) $(call pcmdep,SPECTRUMPAINTER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(SPECTRUMPAINTERDS) $(call dictModule,SPECTRUMPAINTER) -c $(SPECTRUMPAINTERH) $(SPECTRUMPAINTERL)

all-$(MODNAME): $(SPECTRUMPAINTERLIB)
clean-$(MODNAME):
		@rm -f $(SPECTRUMPAINTERO) $(SPECTRUMPAINTERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SPECTRUMPAINTERDEP) $(SPECTRUMPAINTERDS) \
		   $(SPECTRUMPAINTERDH) $(SPECTRUMPAINTERLIB) $(SPECTRUMPAINTERMAP)

distclean::     distclean-$(MODNAME)
