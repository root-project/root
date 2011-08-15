# Module.mk for eve module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers  26/11/2007

MODNAME   := eve
MODDIR    := $(ROOT_SRCDIR)/graf3d/$(MODNAME)
MODDIRS   := $(MODDIR)/src
MODDIRI   := $(MODDIR)/inc

EVEDIR    := $(MODDIR)
EVEDIRS   := $(EVEDIR)/src
EVEDIRI   := $(EVEDIR)/inc

##### libEve #####
EVEL1     := $(MODDIRI)/LinkDef1.h
EVEL2     := $(MODDIRI)/LinkDef2.h
EVEDS1    := $(call stripsrc,$(MODDIRS)/G__Eve1.cxx)
EVEDS2    := $(call stripsrc,$(MODDIRS)/G__Eve2.cxx)
EVEDO1    := $(EVEDS1:.cxx=.o)
EVEDO2    := $(EVEDS2:.cxx=.o)
EVEDH     := $(EVEDS:.cxx=.h)
EVEL      := $(EVEL1) $(EVEL2)
EVEDS     := $(EVEDS1) $(EVEDS2)
EVEDO     := $(EVEDO1) $(EVEDO2)
EVEDH     := $(EVEDS:.cxx=.h)

EVEH1     := TEveBrowser TEveChunkManager TEveCompound \
             TEveElement TEveEventManager TEveGValuators \
             TEveGedEditor TEveMacro TEveManager TEvePad TEveParamList \
             TEveProjectionAxes TEveProjectionBases TEveProjectionManager \
             TEveProjections TEveScene TEveSelection TEveTrans TEveTreeTools \
             TEveUtil TEveVector TEvePathMark TEveVSD TEveViewer TEveWindow \
             TEveSecondarySelectable

EVEH2     := TEveArrow TEveBox TEveCalo \
             TEveDigitSet TEveFrameBox TEveGeo \
             TEveGridStepper TEveLegoEventHandler TEveShape \
             TEveLine TEvePointSet TEvePolygonSetProjected TEveQuadSet \
             TEveRGBAPalette TEveScalableStraightLineSet TEveStraightLineSet \
             TEveText TEveTrack TEveTriangleSet TEveJetCone \
	     TEvePlot3D

EVEH1     := $(foreach stem, $(EVEH1), $(wildcard $(MODDIRI)/$(stem)*.h))
EVEH2     := $(foreach stem, $(EVEH2), $(wildcard $(MODDIRI)/$(stem)*.h))

EVEH      := $(EVEH1) $(EVEH2)
EVES      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
EVEO      := $(call stripsrc,$(EVES:.cxx=.o))

EVEDEP    := $(EVEO:.o=.d) $(EVEDO:.o=.d)

EVELIB    := $(LPATH)/libEve.$(SOEXT)
EVEMAP    := $(EVELIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(EVEH))
ALLLIBS     += $(EVELIB)
ALLMAPS     += $(EVEMAP)

# include all dependency files
INCLUDEFILES += $(EVEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(EVEDIRI)/%.h
		cp $< $@

$(EVELIB):      $(EVEO) $(EVEDO) $(ORDER_) $(MAINLIBS) $(EVELIBDEP) \
                $(FTGLLIB) $(GLEWLIB)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEve.$(SOEXT) $@ "$(EVEO) $(EVEDO)" \
		   "$(EVELIBEXTRA) $(FTGLLIBDIR) $(FTGLLIBS) \
		    $(GLEWLIBDIR) $(GLEWLIBS) $(GLLIBS)"

$(EVEDS1):      $(EVEH1) $(EVEL1) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(EVEH1) $(EVEDIRS)/SolarisCCDictHack.h $(EVEL1)
$(EVEDS2):      $(EVEH2) $(EVEL2) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(EVEH2) $(EVEDIRS)/SolarisCCDictHack.h $(EVEL2)

$(EVEMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(EVEL)
		$(RLIBMAP) -o $@ -l $(EVELIB) \
		   -d $(EVELIBDEPM) -c $(EVEL)

all-$(MODNAME): $(EVELIB) $(EVEMAP)

clean-$(MODNAME):
		@rm -f $(EVEO) $(EVEDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(EVEDEP) $(EVEDS) $(EVEDH) $(EVELIB) $(EVEMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
ifeq ($(ARCH),win32)
$(EVEO) $(EVEDO): CXXFLAGS += $(OPENGLINCDIR:%=-I%) $(FTGLINCDIR:%=-I%) $(FTGLCPPFLAGS)
else
$(EVEO) $(EVEDO): CXXFLAGS += $(OPENGLINCDIR:%=-I%) $(FTGLINCDIR:%=-I%) $(FTGLCPPFLAGS)
$(EVEO): CXXFLAGS += $(GLEWINCDIR:%=-I%) $(GLEWCPPFLAGS)
endif

$(MODNAME)-echo-h1:
	@echo $(EVEH1)

$(MODNAME)-echo-h2:
	@echo $(EVEH2)

# Optimize dictionary with stl containers.
$(EVEDO1): NOOPT = $(OPT)
$(EVEDO2): NOOPT = $(OPT)
