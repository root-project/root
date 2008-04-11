# Module.mk for eg module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := eg
MODDIR       := montecarlo/$(MODNAME)
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

$(EGDS):        $(EGH1) $(EGL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(EGH1) $(EGL)

$(EGMAP):       $(RLIBMAP) $(MAKEFILEDEP) $(EGL)
		$(RLIBMAP) -o $(EGMAP) -l $(EGLIB) \
		   -d $(EGLIBDEPM) -c $(EGL)

all-$(MODNAME): $(EGLIB) $(EGMAP)

clean-$(MODNAME):
		@rm -f $(EGO) $(EGDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(EGDEP) $(EGDS) $(EGDH) $(EGLIB) $(EGMAP)

distclean::     distclean-$(MODNAME)
