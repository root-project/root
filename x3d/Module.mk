# Module.mk for x3d module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := x3d
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

X3DDIR       := $(MODDIR)
X3DDIRS      := $(X3DDIR)/src
X3DDIRI      := $(X3DDIR)/inc

##### libX3d #####
X3DL         := $(MODDIRI)/LinkDef.h
X3DDS        := $(MODDIRS)/G__X3D.cxx
X3DDO        := $(X3DDS:.cxx=.o)
X3DDH        := $(X3DDS:.cxx=.h)

X3DH1        := $(MODDIRI)/TViewerX3D.h
X3DH2        := $(MODDIRI)/x3d.h
X3DH         := $(X3DH1) $(X3DH2)
X3DS1        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
X3DS2        := $(wildcard $(MODDIRS)/*.c)
X3DO         := $(X3DS1:.cxx=.o) $(X3DS2:.c=.o)

X3DDEP       := $(X3DO:.o=.d) $(X3DDO:.o=.d)

X3DLIB       := $(LPATH)/libX3d.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(X3DH))
ALLLIBS     += $(X3DLIB)

# include all dependency files
INCLUDEFILES += $(X3DDEP)

##### local rules #####
include/%.h:    $(X3DDIRI)/%.h
		cp $< $@

$(X3DLIB):      $(X3DO) $(X3DDO) $(MAINLIBS) $(X3DLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libX3d.$(SOEXT) $@ "$(X3DO) $(X3DDO)" \
		   "$(X3DLIBEXTRA) $(XLIBS)"

$(X3DDS):       $(X3DH1) $(X3DL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(X3DH1) $(X3DL)

$(X3DDO):       $(X3DDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-x3d:        $(X3DLIB)

map-x3d:        $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(X3DLIB) \
		   -d $(X3DLIBDEP) -c $(X3DL)

map::           map-x3d

clean-x3d:
		@rm -f $(X3DO) $(X3DDO)

clean::         clean-x3d

distclean-x3d:  clean-x3d
		@rm -f $(X3DDEP) $(X3DDS) $(X3DDH) $(X3DLIB)

distclean::     distclean-x3d
