# Module.mk for eg module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODDIR       := eg
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

EGDIR        := $(MODDIR)
EGDIRS       := $(EGDIR)/src
EGDIRI       := $(EGDIR)/inc

##### libEG #####
EGL          := $(MODDIRI)/LinkDef.h
EGDS         := $(MODDIRS)/G__EG.cxx
EGDO         := $(EGDS:.cxx=.o)
EGDH         := $(EGDS:.cxx=.h)

EGH1         := $(wildcard $(MODDIRI)/T*.h)
EGH          := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
EGS          := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
EGO          := $(EGS:.cxx=.o)

EGDEP        := $(EGO:.o=.d) $(EGDO:.o=.d)

EGLIB        := $(LPATH)/libEG.$(SOEXT)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(EGH))
ALLLIBS     += $(EGLIB)

# include all dependency files
INCLUDEFILES += $(EGDEP)

##### local rules #####
include/%.h:    $(EGDIRI)/%.h
		cp $< $@

$(EGLIB):       $(EGO) $(EGDO) $(MAINLIBS) $(EGLIBDEP)
ifeq ($(PLATFORM),macosx)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEG.$(SOEXT) $@ "$(EGO) $(EGDO)" \
		   "$(EGLIBEXTRA) $(FPYTHIA6LIBDIR) $(FPYTHIA6LIB)"
else
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEG.$(SOEXT) $@ "$(EGO) $(EGDO)" \
		   "$(EGLIBEXTRA)"
endif

$(EGDS):        $(EGH1) $(EGL) $(ROOTCINTTMP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(EGH1) $(EGL)

$(EGDO):        $(EGDS)
		$(CXX) $(NOOPT) $(CXXFLAGS) -I. -o $@ -c $<

all-eg:         $(EGLIB)

map-eg:         $(RLIBMAP)
		$(RLIBMAP) -r $(ROOTMAP) -l $(EGLIB) \
		   -d $(EGLIBDEP) -c $(EGL)

map::           map-eg

clean-eg:
		@rm -f $(EGO) $(EGDO)

clean::         clean-eg

distclean-eg:   clean-eg
		@rm -f $(EGDEP) $(EGDS) $(EGDH) $(EGLIB)

distclean::     distclean-eg
