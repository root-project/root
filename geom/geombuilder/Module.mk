# Module.mk for geombuilder module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME          := geombuilder
MODDIR           := $(ROOT_SRCDIR)/geom/$(MODNAME)
MODDIRS          := $(MODDIR)/src
MODDIRI          := $(MODDIR)/inc

GEOMBUILDERDIR   := $(MODDIR)
GEOMBUILDERDIRS  := $(GEOMBUILDERDIR)/src
GEOMBUILDERDIRI  := $(GEOMBUILDERDIR)/inc

##### libGeomBuilder #####
GEOMBUILDERL     := $(MODDIRI)/LinkDef.h
GEOMBUILDERDS    := $(call stripsrc,$(MODDIRS)/G__GeomBuilder.cxx)
GEOMBUILDERDO    := $(GEOMBUILDERDS:.cxx=.o)
GEOMBUILDERDH    := $(GEOMBUILDERDS:.cxx=.h)

GEOMBUILDERH     := TGeoVolumeEditor.h TGeoBBoxEditor.h TGeoMediumEditor.h \
                    TGeoNodeEditor.h TGeoMatrixEditor.h TGeoManagerEditor.h \
                    TGeoTubeEditor.h TGeoConeEditor.h TGeoTrd1Editor.h \
                    TGeoTrd2Editor.h TGeoMaterialEditor.h TGeoTabManager.h \
                    TGeoSphereEditor.h TGeoPconEditor.h TGeoParaEditor.h \
                    TGeoTorusEditor.h TGeoEltuEditor.h TGeoHypeEditor.h \
                    TGeoPgonEditor.h TGeoTrapEditor.h TGeoGedFrame.h
GEOMBUILDERH     := $(patsubst %,$(MODDIRI)/%,$(GEOMBUILDERH))
GEOMBUILDERS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEOMBUILDERO     := $(call stripsrc,$(GEOMBUILDERS:.cxx=.o))

GEOMBUILDERDEP   := $(GEOMBUILDERO:.o=.d) $(GEOMBUILDERDO:.o=.d)

GEOMBUILDERLIB   := $(LPATH)/libGeomBuilder.$(SOEXT)
GEOMBUILDERMAP   := $(GEOMBUILDERLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS          += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEOMBUILDERH))
ALLLIBS          += $(GEOMBUILDERLIB)
ALLMAPS          += $(GEOMBUILDERMAP)

# include all dependency files
INCLUDEFILES     += $(GEOMBUILDERDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GEOMBUILDERDIRI)/%.h
		cp $< $@

$(GEOMBUILDERLIB): $(GEOMBUILDERO) $(GEOMBUILDERDO) $(ORDER_) $(MAINLIBS) \
                   $(GEOMBUILDERLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeomBuilder.$(SOEXT) $@ \
		   "$(GEOMBUILDERO) $(GEOMBUILDERDO)" \
		   "$(GEOMBUILDERLIBEXTRA)"

$(call pcmrule,GEOMBUILDER)
	$(noop)

$(GEOMBUILDERDS): $(GEOMBUILDERH) $(GEOMBUILDERL) $(ROOTCLINGEXE) $(call pcmdep,GEOMBUILDER)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GEOMBUILDER) -c $(GEOMBUILDERH) $(GEOMBUILDERL)

$(GEOMBUILDERMAP): $(GEOMBUILDERH) $(GEOMBUILDERL) $(ROOTCLINGEXE) $(call pcmdep,GEOMBUILDER)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GEOMBUILDERDS) $(call dictModule,GEOMBUILDER) -c $(GEOMBUILDERH) $(GEOMBUILDERL)

all-$(MODNAME): $(GEOMBUILDERLIB)
clean-$(MODNAME):
		@rm -f $(GEOMBUILDERO) $(GEOMBUILDERDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GEOMBUILDERDEP) $(GEOMBUILDERDS) $(GEOMBUILDERDH) \
		   $(GEOMBUILDERLIB) $(GEOMBUILDERMAP)

distclean::     distclean-$(MODNAME)
