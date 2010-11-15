# Module.mk for gl module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 26/8/2009

MODNAME      := gviz3d
MODDIR       := $(ROOT_SRCDIR)/graf3d/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GVIZ3DDIR    := $(MODDIR)
GVIZ3DDIRS   := $(GVIZ3DDIR)/src
GVIZ3DDIRI   := $(GVIZ3DDIR)/inc

##### libGviz3d #####
GVIZ3DL      := $(MODDIRI)/LinkDef.h
GVIZ3DDS     := $(call stripsrc,$(MODDIRS)/G__Gviz3d.cxx)
GVIZ3DDO     := $(GVIZ3DDS:.cxx=.o)
GVIZ3DDH     := $(GVIZ3DDS:.cxx=.h)

GVIZ3DH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GVIZ3DS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GVIZ3DO      := $(call stripsrc,$(GVIZ3DS:.cxx=.o))

GVIZ3DDEP    := $(GVIZ3DO:.o=.d) $(GVIZ3DDO:.o=.d)

GVIZ3DLIB    := $(LPATH)/libGviz3d.$(SOEXT)
GVIZ3DMAP    := $(GVIZ3DLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GVIZ3DH))
ALLLIBS     += $(GVIZ3DLIB)
ALLMAPS     += $(GVIZ3DMAP)

# include all dependency files
INCLUDEFILES += $(GVIZ3DDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GVIZ3DDIRI)/%.h
		cp $< $@

$(GVIZ3DLIB):   $(GVIZ3DO) $(GVIZ3DDO) $(ORDER_) $(MAINLIBS) $(GVIZ3DLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGviz3d.$(SOEXT) $@ "$(GVIZ3DO) $(GVIZ3DDO)" \
		   "$(GVIZ3DLIBEXTRA)"

$(GVIZ3DDS):    $(GVIZ3DH) $(GVIZ3DL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GVIZ3DH) $(GVIZ3DL)

$(GVIZ3DMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(GVIZ3DL)
		$(RLIBMAP) -o $@ -l $(GVIZ3DLIB) \
		   -d $(GVIZ3DLIBDEPM) -c $(GVIZ3DL)

all-$(MODNAME): $(GVIZ3DLIB) $(GVIZ3DMAP)

clean-$(MODNAME):
		@rm -f $(GVIZ3DO) $(GVIZ3DDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GVIZ3DDEP) $(GVIZ3DDS) $(GVIZ3DDH) $(GVIZ3DLIB) $(GVIZ3DMAP)

distclean::     distclean-$(MODNAME)

