# Module.mk for geompainter module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := geompainter
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GEOMPAINTERDIR  := $(MODDIR)
GEOMPAINTERDIRS := $(GEOMPAINTERDIR)/src
GEOMPAINTERDIRI := $(GEOMPAINTERDIR)/inc

##### libGeomPainter #####
GEOMPAINTERL  := $(MODDIRI)/LinkDef.h
GEOMPAINTERDS := $(MODDIRS)/G__GeomPainter.cxx
GEOMPAINTERDO := $(GEOMPAINTERDS:.cxx=.o)
GEOMPAINTERDH := $(GEOMPAINTERDS:.cxx=.h)

GEOMPAINTERH1 := $(wildcard $(MODDIRI)/T*.h)
GEOMPAINTERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GEOMPAINTERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEOMPAINTERO  := $(GEOMPAINTERS:.cxx=.o)

GEOMPAINTERDEP := $(GEOMPAINTERO:.o=.d) $(GEOMPAINTERDO:.o=.d)

GEOMPAINTERLIB := $(LPATH)/libGeomPainter.$(SOEXT)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEOMPAINTERH))
ALLLIBS       += $(GEOMPAINTERLIB)

# include all dependency files
INCLUDEFILES += $(GEOMPAINTERDEP)

##### local rules #####
include/%.h:    $(GEOMPAINTERDIRI)/%.h
		cp $< $@

$(GEOMPAINTERLIB): $(GEOMPAINTERO) $(GEOMPAINTERDO) $(MAINLIBS) \
                   $(GEOMPAINTERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeomPainter.$(SOEXT) $@ \
		   "$(GEOMPAINTERO) $(GEOMPAINTERDO)" \
		   "$(GEOMPAINTERLIBEXTRA)"

$(GEOMPAINTERDS): $(GEOMPAINTERH1) $(GEOMPAINTERL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEOMPAINTERH1) $(GEOMPAINTERL)

$(GEOMPAINTERDO): $(GEOMPAINTERDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-geompainter: $(GEOMPAINTERLIB)

map-geompainter: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GEOMPAINTERLIB) \
		   -d $(GEOMPAINTERLIBDEP) -c $(GEOMPAINTERL)

map::           map-geompainter

clean-geompainter:
		@rm -f $(GEOMPAINTERO) $(GEOMPAINTERDO)

clean::         clean-geompainter

distclean-geompainter: clean-geompainter
		@rm -f $(GEOMPAINTERDEP) $(GEOMPAINTERDS) $(GEOMPAINTERDH) \
		   $(GEOMPAINTERLIB)

distclean::     distclean-geompainter
