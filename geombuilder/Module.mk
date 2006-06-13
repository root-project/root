# Module.mk for geombuilder module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR           := geombuilder
MODDIRS          := $(MODDIR)/src
MODDIRI          := $(MODDIR)/inc

GEOMBUILDERDIR   := $(MODDIR)
GEOMBUILDERDIRS  := $(GEOMBUILDERDIR)/src
GEOMBUILDERDIRI  := $(GEOMBUILDERDIR)/inc

##### libGeomBuilder #####
GEOMBUILDERL     := $(MODDIRI)/LinkDef.h
GEOMBUILDERDS    := $(MODDIRS)/G__GeomBuilder.cxx
GEOMBUILDERDO    := $(GEOMBUILDERDS:.cxx=.o)
GEOMBUILDERDH    := $(GEOMBUILDERDS:.cxx=.h)

GEOMBUILDERH     := TGeoVolumeEditor.h TGeoBBoxEditor.h TGeoMediumEditor.h \
                    TGeoNodeEditor.h TGeoMatrixEditor.h TGeoManagerEditor.h \
                    TGeoTubeEditor.h TGeoConeEditor.h TGeoTrd1Editor.h \
                    TGeoTrd2Editor.h TGeoMaterialEditor.h TGeoTabManager.h \
                    TGeoSphereEditor.h
GEOMBUILDERH     := $(patsubst %,$(MODDIRI)/%,$(GEOMBUILDERH))
GEOMBUILDERS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEOMBUILDERO     := $(GEOMBUILDERS:.cxx=.o)

GEOMBUILDERDEP   := $(GEOMBUILDERO:.o=.d) $(GEOMBUILDERDO:.o=.d)

GEOMBUILDERLIB   := $(LPATH)/libGeomBuilder.$(SOEXT)

# used in the main Makefile
ALLHDRS          += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEOMBUILDERH))
ALLLIBS          += $(GEOMBUILDERLIB)

# include all dependency files
INCLUDEFILES     += $(GEOMBUILDERDEP)

##### local rules #####
include/%.h:    $(GEOMBUILDERDIRI)/%.h
		cp $< $@

$(GEOMBUILDERLIB): $(GEOMBUILDERO) $(GEOMBUILDERDO) $(ORDER_) $(MAINLIBS) \
                   $(GEOMBUILDERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeomBuilder.$(SOEXT) $@ \
		   "$(GEOMBUILDERO) $(GEOMBUILDERDO)" \
		   "$(GEOMBUILDERLIBEXTRA)"

$(GEOMBUILDERDS): $(GEOMBUILDERH) $(GEOMBUILDERL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEOMBUILDERH) $(GEOMBUILDERL)

all-geombuilder: $(GEOMBUILDERLIB)

map-geombuilder: $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(GEOMBUILDERLIB) \
		   -d $(GEOMBUILDERLIBDEP) -c $(GEOMBUILDERL)

map::           map-geombuilder

clean-geombuilder:
		@rm -f $(GEOMBUILDERO) $(GEOMBUILDERDO)

clean::         clean-geombuilder

distclean-geombuilder: clean-geombuilder
		@rm -f $(GEOMBUILDERDEP) $(GEOMBUILDERDS) $(GEOMBUILDERDH) \
		   $(GEOMBUILDERLIB)

distclean::     distclean-geombuilder
