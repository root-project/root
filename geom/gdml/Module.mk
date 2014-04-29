# Module.mk for gdml module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Ben Lloyd 09/11/06

MODNAME      := gdml
MODDIR       := $(ROOT_SRCDIR)/geom/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GDMLDIR      := $(MODDIR)
GDMLDIRS     := $(GDMLDIR)/src
GDMLDIRI     := $(GDMLDIR)/inc

##### libGdml #####
GDMLL        := $(MODDIRI)/LinkDef.h
GDMLDS       := $(call stripsrc,$(MODDIRS)/G__Gdml.cxx)
GDMLDO       := $(GDMLDS:.cxx=.o)
GDMLDH       := $(GDMLDS:.cxx=.h)

GDMLH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GDMLS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GDMLO        := $(call stripsrc,$(GDMLS:.cxx=.o))

GDMLDEP      := $(GDMLO:.o=.d) $(GDMLDO:.o=.d)

GDMLLIB      := $(LPATH)/libGdml.$(SOEXT)
GDMLMAP      := $(GDMLLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GDMLH))
ALLLIBS      += $(GDMLLIB)
ALLMAPS      += $(GDMLMAP)

# include all dependency files
INCLUDEFILES += $(GDMLDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GDMLDIRI)/%.h
		cp $< $@

$(GDMLLIB):     $(GDMLO) $(GDMLDO) $(ORDER_) $(MAINLIBS) $(GDMLLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGdml.$(SOEXT) $@ "$(GDMLO) $(GDMLDO)" \
		   "$(GDMLLIBEXTRA)"

$(call pcmrule,GDML)
	$(noop)

$(GDMLDS):      $(GDMLH) $(GDMLL) $(ROOTCLINGEXE) $(call pcmdep,GDML)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GDML) -c $(GDMLH) $(GDMLL)

$(GDMLMAP):     $(GDMLH) $(GDMLL) $(ROOTCLINGEXE) $(call pcmdep,GDML)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GDMLDS) $(call dictModule,GDML) -c $(GDMLH) $(GDMLL)

all-$(MODNAME): $(GDMLLIB)

clean-$(MODNAME):
		@rm -f $(GDMLO) $(GDMLDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GDMLDEP) $(GDMLDS) $(GDMLDH) $(GDMLLIB) $(GDMLMAP)

distclean::     distclean-$(MODNAME)
