# Module.mk for geocad module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Cinzia Luzzi  2/7/2012

MODNAME      := geocad
MODDIR       := $(ROOT_SRCDIR)/geom/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc
 
GEOCADDIR    := $(MODDIR)
GEOCADDIRS   := $(GEOCADDIR)/src
GEOCADDIRI   := $(GEOCADDIR)/inc

##### libGeoCad #####
GEOCADL      := $(MODDIRI)/LinkDef.h
GEOCADDS     := $(call stripsrc,$(MODDIRS)/G__GeoCad.cxx)
GEOCADDO     := $(GEOCADDS:.cxx=.o)
GEOCADDH     := $(GEOCADDS:.cxx=.h)

GEOCADH1     := RootOCC.h OCCStep.h
GEOCADH2     := TRootStep.h
GEOCADH1     := $(patsubst %,$(MODDIRI)/%,$(GEOCADH1))
GEOCADH2     := $(patsubst %,$(MODDIRI)/%,$(GEOCADH2))
GEOCADH      := $(GEOCADH1) $(GEOCADH2)
GEOCADS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GEOCADO      := $(call stripsrc,$(GEOCADS:.cxx=.o))

GEOCADDEP    := $(GEOCADO:.o=.d) $(GEOCADDO:.o=.d)

GEOCADLIB    := $(LPATH)/libGeoCad.$(SOEXT)
GEOCADMAP    := $(GEOCADLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GEOCADH))
ALLLIBS     += $(GEOCADLIB)
ALLMAPS     += $(GEOCADMAP)

# include all dependency files
INCLUDEFILES += $(GEOCADDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GEOCADDIRI)/%.h
		cp $< $@

$(GEOCADLIB):   $(GEOCADO) $(GEOCADDO) $(ORDER_) $(MAINLIBS) $(GEOCADLIBDEP)
		$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGeoCad.$(SOEXT) $@ "$(GEOCADO) $(GEOCADDO)" \
		   "$(GEOCADLIBEXTRA) $(OCCLIBDIR) $(OCCLIB)"

$(GEOCADDS):    $(GEOCADH1) $(GEOCADH2) $(GEOCADL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GEOCADH2) $(GEOCADL)

$(GEOCADMAP):   $(RLIBMAP) $(MAKEFILEDEP) $(GEOCADL) 
		$(RLIBMAP) -o $@ -l $(GEOCADLIB) \
		   -d $(GEOCADLIBDEPM) -c $(GEOCADL)

all-$(MODNAME): $(GEOCADLIB) $(GEOCADMAP)

clean-$(MODNAME):
		@rm -f $(GEOCADO) $(GEOCADDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GEOCADDEP) $(GEOCADDS) $(GEOCADDH) $(GEOCADLIB) $(GEOCADMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GEOCADO): CXXFLAGS += $(OCCINCDIR:%=-I%)
$(GEOCADO) $(GEOCADDO): CXXFLAGS := $(filter-out -Wshadow,$(CXXFLAGS))
$(GEOCADO) $(GEOCADDO): CXXFLAGS := $(filter-out -Woverloaded-virtual,$(CXXFLAGS))
$(GEOCADO) $(GEOCADDO): CXXFLAGS := $(filter-out -Wall,$(CXXFLAGS))
$(GEOCADO) $(GEOCADDO): CXXFLAGS := $(filter-out -W,$(CXXFLAGS))
