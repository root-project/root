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
GEOMPAINTERMAP := $(GEOMPAINTERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS       += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEOMPAINTERH))
ALLLIBS       += $(GEOMPAINTERLIB)
ALLMAPS       += $(GEOMPAINTERMAP)

# include all dependency files
INCLUDEFILES += $(GEOMPAINTERDEP)

##### local rules #####
include/%.h:    $(GEOMPAINTERDIRI)/%.h
		cp $< $@

$(GEOMPAINTERLIB): $(GEOMPAINTERO) $(GEOMPAINTERDO) $(ORDER_) $(MAINLIBS) \
                   $(GEOMPAINTERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeomPainter.$(SOEXT) $@ \
		   "$(GEOMPAINTERO) $(GEOMPAINTERDO)" \
		   "$(GEOMPAINTERLIBEXTRA)"

$(GEOMPAINTERDS): $(GEOMPAINTERH1) $(GEOMPAINTERL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEOMPAINTERH1) $(GEOMPAINTERL)

$(GEOMPAINTERMAP): $(RLIBMAP) $(MAKEFILEDEP) $(GEOMPAINTERL)
		$(RLIBMAP) -o $(GEOMPAINTERMAP) -l $(GEOMPAINTERLIB) \
		   -d $(GEOMPAINTERLIBDEPM) -c $(GEOMPAINTERL)

all-geompainter: $(GEOMPAINTERLIB) $(GEOMPAINTERMAP)

clean-geompainter:
		@rm -f $(GEOMPAINTERO) $(GEOMPAINTERDO)

clean::         clean-geompainter

distclean-geompainter: clean-geompainter
		@rm -f $(GEOMPAINTERDEP) $(GEOMPAINTERDS) $(GEOMPAINTERDH) \
		   $(GEOMPAINTERLIB) $(GEOMPAINTERMAP)

distclean::     distclean-geompainter
