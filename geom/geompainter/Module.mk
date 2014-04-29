# Module.mk for geompainter module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := geompainter
MODDIR       := $(ROOT_SRCDIR)/geom/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GEOMPAINTERDIR  := $(MODDIR)
GEOMPAINTERDIRS := $(GEOMPAINTERDIR)/src
GEOMPAINTERDIRI := $(GEOMPAINTERDIR)/inc

##### libGeomPainter #####
GEOMPAINTERL  := $(MODDIRI)/LinkDef.h
GEOMPAINTERDS := $(call stripsrc,$(MODDIRS)/G__GeomPainter.cxx)
GEOMPAINTERDO := $(GEOMPAINTERDS:.cxx=.o)
GEOMPAINTERDH := $(GEOMPAINTERDS:.cxx=.h)

GEOMPAINTERH1 := $(wildcard $(MODDIRI)/T*.h)
GEOMPAINTERH  := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GEOMPAINTERS  := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEOMPAINTERO  := $(call stripsrc,$(GEOMPAINTERS:.cxx=.o))

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
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GEOMPAINTERDIRI)/%.h
		cp $< $@

$(GEOMPAINTERLIB): $(GEOMPAINTERO) $(GEOMPAINTERDO) $(ORDER_) $(MAINLIBS) \
                   $(GEOMPAINTERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeomPainter.$(SOEXT) $@ \
		   "$(GEOMPAINTERO) $(GEOMPAINTERDO)" \
		   "$(GEOMPAINTERLIBEXTRA)"

$(call pcmrule,GEOMPAINTER)
	$(noop)

$(GEOMPAINTERDS): $(GEOMPAINTERH1) $(GEOMPAINTERL) $(ROOTCLINGEXE) $(call pcmdep,GEOMPAINTER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GEOMPAINTER) -c $(GEOMPAINTERH1) $(GEOMPAINTERL)

$(GEOMPAINTERMAP): $(GEOMPAINTERH1) $(GEOMPAINTERL) $(ROOTCLINGEXE) $(call pcmdep,GEOMPAINTER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GEOMPAINTERDS) $(call dictModule,GEOMPAINTER) -c $(GEOMPAINTERH1) $(GEOMPAINTERL)

all-$(MODNAME): $(GEOMPAINTERLIB)
clean-$(MODNAME):
		@rm -f $(GEOMPAINTERO) $(GEOMPAINTERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GEOMPAINTERDEP) $(GEOMPAINTERDS) $(GEOMPAINTERDH) \
		   $(GEOMPAINTERLIB) $(GEOMPAINTERMAP)

distclean::     distclean-$(MODNAME)
