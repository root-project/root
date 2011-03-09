# Module.mk for geom module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := geom
MODDIR       := $(ROOT_SRCDIR)/geom/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GEOMDIR      := $(MODDIR)
GEOMDIRS     := $(GEOMDIR)/src
GEOMDIRI     := $(GEOMDIR)/inc

##### libGeom #####
GEOML1       := $(MODDIRI)/LinkDef1.h
GEOML2       := $(MODDIRI)/LinkDef2.h
GEOMDS1      := $(call stripsrc,$(MODDIRS)/G__Geom1.cxx)
GEOMDS2      := $(call stripsrc,$(MODDIRS)/G__Geom2.cxx)
GEOMDO1      := $(GEOMDS1:.cxx=.o)
GEOMDO2      := $(GEOMDS2:.cxx=.o)
GEOMDS       := $(GEOMDS1) $(GEOMDS2)
GEOMDO       := $(GEOMDO1) $(GEOMDO2)
GEOMDH       := $(GEOMDS:.cxx=.h)

GEOMH1       := TGeoAtt.h TGeoBoolNode.h \
                TGeoMedium.h TGeoMaterial.h \
                TGeoMatrix.h TGeoVolume.h TGeoNode.h \
                TGeoVoxelFinder.h TGeoShape.h TGeoBBox.h \
                TGeoPara.h TGeoTube.h TGeoTorus.h TGeoSphere.h \
                TGeoEltu.h TGeoHype.h TGeoCone.h TGeoPcon.h \
                TGeoPgon.h TGeoArb8.h TGeoTrd1.h TGeoTrd2.h \
                TGeoManager.h TGeoCompositeShape.h TGeoShapeAssembly.h \
                TGeoScaledShape.h TVirtualGeoPainter.h TVirtualGeoTrack.h \
		TGeoPolygon.h TGeoXtru.h TGeoPhysicalNode.h \
                TGeoHelix.h TGeoParaboloid.h TGeoElement.h TGeoHalfSpace.h \
                TGeoBuilder.h TGeoNavigator.h
GEOMH2       := TGeoPatternFinder.h TGeoCache.h TVirtualMagField.h \
                TGeoUniformMagField.h TGeoGlobalMagField.h TGeoBranchArray.h
GEOMH1       := $(patsubst %,$(MODDIRI)/%,$(GEOMH1))
GEOMH2       := $(patsubst %,$(MODDIRI)/%,$(GEOMH2))
GEOMH        := $(GEOMH1) $(GEOMH2)
GEOMS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEOMO        := $(call stripsrc,$(GEOMS:.cxx=.o))

GEOMDEP      := $(GEOMO:.o=.d) $(GEOMDO:.o=.d)

GEOMLIB      := $(LPATH)/libGeom.$(SOEXT)
GEOMMAP      := $(GEOMLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEOMH))
ALLLIBS     += $(GEOMLIB)
ALLMAPS     += $(GEOMMAP)

# include all dependency files
INCLUDEFILES += $(GEOMDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GEOMDIRI)/%.h
		cp $< $@

$(GEOMLIB):     $(GEOMO) $(GEOMDO) $(ORDER_) $(MAINLIBS) $(GEOMLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeom.$(SOEXT) $@ "$(GEOMO) $(GEOMDO)" \
		   "$(GEOMLIBEXTRA)"

$(GEOMDS1):     $(GEOMH1) $(GEOML1) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEOMH1) $(GEOML1)

$(GEOMDS2):     $(GEOMH2) $(GEOML2) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEOMH2) $(GEOML2)

$(GEOMMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(GEOML1) $(GEOML2)
		$(RLIBMAP) -o $@ -l $(GEOMLIB) \
		   -d $(GEOMLIBDEPM) -c $(GEOML1) $(GEOML2)

all-$(MODNAME): $(GEOMLIB) $(GEOMMAP)

clean-$(MODNAME):
		@rm -f $(GEOMO) $(GEOMDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GEOMDEP) $(GEOMDS) $(GEOMDH) $(GEOMLIB) $(GEOMMAP)

distclean::     distclean-$(MODNAME)
