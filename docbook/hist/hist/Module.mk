# Module.mk for hist module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := hist
MODDIR       := $(ROOT_SRCDIR)/hist/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

HISTDIR      := $(MODDIR)
HISTDIRS     := $(HISTDIR)/src
HISTDIRI     := $(HISTDIR)/inc

##### libHist #####
HISTL        := $(MODDIRI)/LinkDef.h
HISTDS       := $(call stripsrc,$(MODDIRS)/G__Hist.cxx)
HISTDO       := $(HISTDS:.cxx=.o)
HISTDH       := $(HISTDS:.cxx=.h)

HISTH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
HISTHMAT     := $(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h))
#HISTHMAT     += mathcore/inc/Math/WrappedFunction.h
HISTHH       := $(HISTH) $(HISTHMAT) 

HISTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HISTO        := $(call stripsrc,$(HISTS:.cxx=.o))

HISTDEP      := $(HISTO:.o=.d) $(HISTDO:.o=.d)

HISTLIB      := $(LPATH)/libHist.$(SOEXT)
HISTMAP      := $(HISTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(HISTHH))
#ALLHDRS     += $(patsubst $(MODDIRI)/Math/%.h,include/Math/%.h,$(HISTHH))
ALLLIBS     += $(HISTLIB)
ALLMAPS     += $(HISTMAP)

# include all dependency files
INCLUDEFILES += $(HISTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/Math/%.h: $(HISTDIRI)/Math/%.h
		@(if [ ! -d "include/Math" ]; then     \
		   mkdir -p include/Math;              \
		fi)
		cp $< $@

include/%.h:    $(HISTDIRI)/%.h
		cp $< $@

$(HISTLIB):     $(HISTO) $(HISTDO) $(ORDER_) $(MAINLIBS) $(HISTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHist.$(SOEXT) $@ "$(HISTO) $(HISTDO)" \
		   "$(HISTLIBEXTRA)"

$(HISTDS):      $(HISTHH) $(HISTL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(HISTHH) $(HISTL)

$(HISTMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(HISTL)
		$(RLIBMAP) -o $@ -l $(HISTLIB) \
		   -d $(HISTLIBDEPM) -c $(HISTL)

all-$(MODNAME): $(HISTLIB) $(HISTMAP)

clean-$(MODNAME):
		@rm -f $(HISTO) $(HISTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HISTDEP) $(HISTDS) $(HISTDH) $(HISTLIB) $(HISTMAP)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(HISTDO): NOOPT = $(OPT)
