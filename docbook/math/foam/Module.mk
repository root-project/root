# Module.mk for foam module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := foam
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

FOAMDIR      := $(MODDIR)
FOAMDIRS     := $(FOAMDIR)/src
FOAMDIRI     := $(FOAMDIR)/inc

##### libFoam.so #####
FOAML      := $(MODDIRI)/LinkDef.h
FOAMDS     := $(call stripsrc,$(MODDIRS)/G__Foam.cxx)
FOAMDO     := $(FOAMDS:.cxx=.o)
FOAMDH     := $(FOAMDS:.cxx=.h)

FOAMH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
FOAMS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
FOAMO      := $(call stripsrc,$(FOAMS:.cxx=.o))

FOAMDEP    := $(FOAMO:.o=.d) $(FOAMDO:.o=.d)

FOAMLIB    := $(LPATH)/libFoam.$(SOEXT)
FOAMMAP    := $(FOAMLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(FOAMH))
ALLLIBS     += $(FOAMLIB)
ALLMAPS     += $(FOAMMAP)

# include all dependency files
INCLUDEFILES += $(FOAMDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FOAMDIRI)/%.h
		cp $< $@

$(FOAMLIB):     $(FOAMO) $(FOAMDO) $(ORDER_) $(MAINLIBS) $(FOAMLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libFoam.$(SOEXT) $@ "$(FOAMO) $(FOAMDO)" \
		   "$(FOAMLIBEXTRA)"

$(FOAMDS):      $(FOAMH) $(FOAML) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(FOAMH) $(FOAML)

$(FOAMMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(FOAML)
		$(RLIBMAP) -o $@ -l $(FOAMLIB) \
		   -d $(FOAMLIBDEPM) -c $(FOAML)

all-$(MODNAME): $(FOAMLIB) $(FOAMMAP)

clean-$(MODNAME):
		@rm -f $(FOAMO) $(FOAMDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FOAMDEP) $(FOAMDS) $(FOAMDH) $(FOAMLIB) $(FOAMMAP)

distclean::     distclean-$(MODNAME)
