# Module.mk for histpainter module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := histpainter
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HISTPAINTERDIR  := $(MODDIR)
HISTPAINTERDIRS := $(HISTPAINTERDIR)/src
HISTPAINTERDIRI := $(HISTPAINTERDIR)/inc

##### libHistPainter #####
HISTPAINTERL  := $(MODDIRI)/LinkDef.h
HISTPAINTERDS := $(MODDIRS)/G__HistPainter.cxx
HISTPAINTERDO := $(HISTPAINTERDS:.cxx=.o)
HISTPAINTERDH := $(HISTPAINTERDS:.cxx=.h)

HISTPAINTERH1 := $(wildcard $(MODDIRI)/T*.h)
HISTPAINTERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HISTPAINTERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HISTPAINTERO  := $(HISTPAINTERS:.cxx=.o)

HISTPAINTERDEP := $(HISTPAINTERO:.o=.d) $(HISTPAINTERDO:.o=.d)

HISTPAINTERLIB := $(LPATH)/libHistPainter.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(HISTPAINTERH))
ALLLIBS       += $(HISTPAINTERLIB)

# include all dependency files
INCLUDEFILES += $(HISTPAINTERDEP)

##### local rules #####
include/%.h:    $(HISTPAINTERDIRI)/%.h
		cp $< $@

$(HISTPAINTERLIB): $(HISTPAINTERO) $(HISTPAINTERDO) $(MAINLIBS) \
                   $(HISTPAINTERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHistPainter.$(SOEXT) $@ \
		   "$(HISTPAINTERO) $(HISTPAINTERDO)" \
		   "$(HISTPAINTERLIBEXTRA)"

$(HISTPAINTERDS): $(HISTPAINTERH1) $(HISTPAINTERL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(HISTPAINTERH1) $(HISTPAINTERL)

$(HISTPAINTERDO): $(HISTPAINTERDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-histpainter: $(HISTPAINTERLIB)

map-histpainter: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(HISTPAINTERLIB) \
		   -d $(HISTPAINTERLIBDEP) -c $(HISTPAINTERL)

map::           map-histpainter

clean-histpainter:
		@rm -f $(HISTPAINTERO) $(HISTPAINTERDO)

clean::         clean-histpainter

distclean-histpainter: clean-histpainter
		@rm -f $(HISTPAINTERDEP) $(HISTPAINTERDS) $(HISTPAINTERDH) \
		   $(HISTPAINTERLIB)

distclean::     distclean-histpainter
