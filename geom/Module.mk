# Module.mk for geom module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := geom
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GEOMDIR      := $(MODDIR)
GEOMDIRS     := $(GEOMDIR)/src
GEOMDIRI     := $(GEOMDIR)/inc

##### libGeom #####
GEOML        := $(MODDIRI)/LinkDef.h
GEOMDS       := $(MODDIRS)/G__Geom.cxx
GEOMDO       := $(GEOMDS:.cxx=.o)
GEOMDH       := $(GEOMDS:.cxx=.h)

GEOMH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GEOMS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEOMO        := $(GEOMS:.cxx=.o)

GEOMDEP      := $(GEOMO:.o=.d) $(GEOMDO:.o=.d)

GEOMLIB      := $(LPATH)/libGeom.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEOMH))
ALLLIBS     += $(GEOMLIB)

# include all dependency files
INCLUDEFILES += $(GEOMDEP)

##### local rules #####
include/%.h:    $(GEOMDIRI)/%.h
		cp $< $@

$(GEOMLIB):     $(GEOMO) $(GEOMDO) $(MAINLIBS) $(GEOMLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeom.$(SOEXT) $@ "$(GEOMO) $(GEOMDO)" \
		   "$(GEOMLIBEXTRA)"

$(GEOMDS):      $(GEOMH) $(GEOML) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEOMH) $(GEOML)

$(GEOMDO):      $(GEOMDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-geom:       $(GEOMLIB)

clean-geom:
		@rm -f $(GEOMO) $(GEOMDO)

clean::         clean-geom

distclean-geom: clean-geom
		@rm -f $(GEOMDEP) $(GEOMDS) $(GEOMDH) $(GEOMLIB)

distclean::     distclean-geom
