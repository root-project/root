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
GEOML0       := $(MODDIRI)/LinkDef.h
GEOMLS       := $(MODDIRI)/LinkDef1.h $(MODDIRI)/LinkDef2.h
GEOMDS       := $(call stripsrc,$(MODDIRS)/G__Geom.cxx)
GEOMDO       := $(GEOMDS:.cxx=.o)
GEOMDH       := $(GEOMDS:.cxx=.h)

GEOMH1       := TGeoAtt.h TGeoStateInfo.h TGeoBoolNode.h \
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
                TGeoUniformMagField.h TGeoGlobalMagField.h TGeoBranchArray.h \
                TGeoExtension.h TGeoParallelWorld.h
GEOMH3       := TGeoRCPtr.h
GEOMH1       := $(patsubst %,$(MODDIRI)/%,$(GEOMH1))
GEOMH2       := $(patsubst %,$(MODDIRI)/%,$(GEOMH2))
GEOMH3       := $(patsubst %,$(MODDIRI)/%,$(GEOMH3))
GEOMH        := $(GEOMH1) $(GEOMH2) $(GEOMH3)
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
		   "$(GEOMLIBEXTRA) $(OSTHREADLIBDIR) $(OSTHREADLIB)"

$(call pcmrule,GEOM)
	$(noop)

$(GEOMDS):      $(GEOMH) $(GEOML0) $(GEOMLS) $(ROOTCLINGEXE) $(call pcmdep,GEOM)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GEOM) -c -I$(ROOT_SRCDIR) $(GEOMH1) $(GEOMH2) $(GEOML0)

$(GEOMMAP):     $(GEOMH) $(GEOML0) $(GEOMLS) $(ROOTCLINGEXE) $(call pcmdep,GEOM)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GEOMDS) $(call dictModule,GEOM) -c -I$(ROOT_SRCDIR) $(GEOMH1) $(GEOMH2) $(GEOML0)

all-$(MODNAME): $(GEOMLIB)

clean-$(MODNAME):
		@rm -f $(GEOMO) $(GEOMDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GEOMDEP) $(GEOMDS) $(GEOMDH) $(GEOMLIB) $(GEOMMAP)

distclean::     distclean-$(MODNAME)
