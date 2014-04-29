# Module.mk for eg module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := eg
MODDIR       := $(ROOT_SRCDIR)/montecarlo/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

EGDIR        := $(MODDIR)
EGDIRS       := $(EGDIR)/src
EGDIRI       := $(EGDIR)/inc

##### libEG #####
EGL          := $(MODDIRI)/LinkDef.h
EGDS         := $(call stripsrc,$(MODDIRS)/G__EG.cxx)
EGDO         := $(EGDS:.cxx=.o)
EGDH         := $(EGDS:.cxx=.h)

EGH1         := $(wildcard $(MODDIRI)/T*.h)
EGH          := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
EGS          := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
EGO          := $(call stripsrc,$(EGS:.cxx=.o))

EGDEP        := $(EGO:.o=.d) $(EGDO:.o=.d)

EGLIB        := $(LPATH)/libEG.$(SOEXT)
EGMAP        := $(EGLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(EGH))
ALLLIBS     += $(EGLIB)
ALLMAPS     += $(EGMAP)

# include all dependency files
INCLUDEFILES += $(EGDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(EGDIRI)/%.h
		cp $< $@

$(EGLIB):       $(EGO) $(EGDO) $(ORDER_) $(MAINLIBS) $(EGLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libEG.$(SOEXT) $@ "$(EGO) $(EGDO)" \
		   "$(EGLIBEXTRA)"

$(call pcmrule,EG)
	$(noop)

$(EGDS):        $(EGH1) $(EGL) $(ROOTCLINGEXE) $(call pcmdep,EG)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,EG) -c $(EGH1) $(EGL)

$(EGMAP):       $(EGH1) $(EGL) $(ROOTCLINGEXE) $(call pcmdep,EG)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(EGDS) $(call dictModule,EG) -c $(EGH1) $(EGL)

all-$(MODNAME): $(EGLIB)

clean-$(MODNAME):
		@rm -f $(EGO) $(EGDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(EGDEP) $(EGDS) $(EGDH) $(EGLIB) $(EGMAP)

distclean::     distclean-$(MODNAME)

##### target variables #####
$(MODDIRS)/TGenerator.o: CXXFLAGS:=$(filter-out -Wshadow,$(CXXFLAGS))
