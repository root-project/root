# Module.mk for spectrumpainter module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Olivier Couet, 27/11/2006

MODDIR       := spectrumpainter
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

SPECTRUMPAINTERDIR  := $(MODDIR)
SPECTRUMPAINTERDIRS := $(SPECTRUMPAINTERDIR)/src
SPECTRUMPAINTERDIRI := $(SPECTRUMPAINTERDIR)/inc

##### libSpectrumPainter #####
SPECTRUMPAINTERL  := $(MODDIRI)/LinkDef.h
SPECTRUMPAINTERDS := $(MODDIRS)/G__Spectrum2Painter.cxx
SPECTRUMPAINTERDO := $(SPECTRUMPAINTERDS:.cxx=.o)
SPECTRUMPAINTERDH := $(SPECTRUMPAINTERDS:.cxx=.h)

SPECTRUMPAINTERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SPECTRUMPAINTERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SPECTRUMPAINTERO  := $(SPECTRUMPAINTERS:.cxx=.o)

SPECTRUMPAINTERDEP := $(SPECTRUMPAINTERO:.o=.d) $(SPECTRUMPAINTERDO:.o=.d)

SPECTRUMPAINTERLIB := $(LPATH)/libSpectrumPainter.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SPECTRUMPAINTERH))
ALLLIBS       += $(SPECTRUMPAINTERLIB)

# include all dependency files
INCLUDEFILES += $(SPECTRUMPAINTERDEP)

##### local rules #####
include/%.h:    $(SPECTRUMPAINTERDIRI)/%.h
		cp $< $@

$(SPECTRUMPAINTERLIB): $(SPECTRUMPAINTERO) $(SPECTRUMPAINTERDO) $(ORDER_) $(MAINLIBS) $(SPECTRUMPAINTERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSpectrumPainter.$(SOEXT) $@ \
		   "$(SPECTRUMPAINTERO) $(SPECTRUMPAINTERDO)" \
		   "$(SPECTRUMPAINTERLIBEXTRA)"

$(SPECTRUMPAINTERDS): $(SPECTRUMPAINTERH) $(SPECTRUMPAINTERL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(SPECTRUMPAINTERH) $(SPECTRUMPAINTERL)

all-spectrumpainter: $(SPECTRUMPAINTERLIB)

map-spectrumpainter: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(SPECTRUMPAINTERLIB) \
		   -d $(SPECTRUMPAINTERLIBDEP) -c $(SPECTRUMPAINTERL)

map::           map-spectrumpainter

clean-spectrumpainter:
		@rm -f $(SPECTRUMPAINTERO) $(SPECTRUMPAINTERDO)

clean::         clean-spectrumpainter

distclean-spectrumpainter: clean-spectrumpainter
		@rm -f $(SPECTRUMPAINTERDEP) $(SPECTRUMPAINTERDS) $(SPECTRUMPAINTERDH) \
		   $(SPECTRUMPAINTERLIB)

distclean::     distclean-spectrumpainter
