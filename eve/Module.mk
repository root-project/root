# Module.mk for eve module
# Copyright (c) 2007 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers  26/11/2007

MODDIR    := eve
MODDIRS   := $(MODDIR)/src
MODDIRI   := $(MODDIR)/inc

EVEDIR    := $(MODDIR)
EVEDIRS   := $(EVEDIR)/src
EVEDIRI   := $(EVEDIR)/inc

##### libEve #####
EVEL      := $(MODDIRI)/LinkDef.h
EVEDS     := $(MODDIRS)/G__Eve.cxx
EVEDO     := $(EVEDS:.cxx=.o)
EVEDH     := $(EVEDS:.cxx=.h)

EVEH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
EVES      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
EVEO      := $(EVES:.cxx=.o)

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
include/%.h:    $(EVEDIRI)/%.h
		cp $< $@

$(EVELIB):      $(EVEO) $(EVEDO) $(ORDER_) $(MAINLIBS) $(EVELIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEve.$(SOEXT) $@ "$(EVEO) $(EVEDO)" \
		   "$(EVELIBEXTRA) $(FTGLLIBDIR) $(FTGLLIBS) $(GLLIBS)"

$(EVEDS):       $(EVEH) $(EVEL) $(ROOTCINTTMPEXE)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(EVEH) $(EVEDIRS)/SolarisCCDictHack.h $(EVEL)

$(EVEMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(EVEL)
		$(RLIBMAP) -o $(EVEMAP) -l $(EVELIB) \
		   -d $(EVELIBDEPM) -c $(EVEL)

all-eve:        $(EVELIB) $(EVEMAP)

clean-eve:
		@rm -f $(EVEO) $(EVEDO)

clean::         clean-eve

distclean-eve: clean-eve
		@rm -f $(EVEDEP) $(EVEDS) $(EVEDH) $(EVELIB) $(EVEMAP)

distclean::     distclean-eve

##### extra rules ######
ifeq ($(ARCH),win32)
$(EVEO) $(EVEDO): CXXFLAGS += $(OPENGLINCDIR:%=-I%) $(FTGLINCDIR:%=-I%)
else
$(EVEO) $(EVEDO): CXXFLAGS += $(OPENGLINCDIR:%=-I%) $(FTGLINCDIR:%=-I%)
endif
