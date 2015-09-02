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

HISTMH       := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h)) \
		$(filter-out $(MODDIRI)/Math/LinkDef%,$(wildcard $(MODDIRI)/Math/*.h)) \
		$(filter-out $(MODDIRI)/v5/LinkDef%,$(wildcard $(MODDIRI)/v5/*.h))
HISTINCH     := $(patsubst $(MODDIRI)/%,include/%,$(HISTMH))
#HISTHMAT     += mathcore/inc/Math/WrappedFunction.h

HISTS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
HISTO        := $(call stripsrc,$(HISTS:.cxx=.o))

HISTDEP      := $(HISTO:.o=.d) $(HISTDO:.o=.d)

HISTLIB      := $(LPATH)/libHist.$(SOEXT)
HISTMAP      := $(HISTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(HISTINCH)
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

include/v5/%.h: $(HISTDIRI)/v5/%.h
		@(if [ ! -d "include/v5" ]; then     \
		   mkdir -p include/v5;              \
		fi)
		cp $< $@

include/%.h:    $(HISTDIRI)/%.h
		cp $< $@

$(HISTLIB):     $(HISTO) $(HISTDO) $(ORDER_) $(MAINLIBS) $(HISTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libHist.$(SOEXT) $@ "$(HISTO) $(HISTDO)" \
		   "$(HISTLIBEXTRA)"

$(call pcmrule,HIST)
	$(noop)

$(HISTDS):      $(HISTINCH) $(HISTL) $(ROOTCLINGEXE) $(call pcmdep,HIST)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,HIST) -c -writeEmptyRootPCM $(patsubst include/%,%,$(HISTINCH)) $(HISTL)

$(HISTMAP):     $(HISTINCH) $(HISTL) $(ROOTCLINGEXE) $(call pcmdep,HIST)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(HISTDS) $(call dictModule,HIST) -c $(patsubst include/%,%,$(HISTINCH)) $(HISTL)

all-$(MODNAME): $(HISTLIB)

clean-$(MODNAME):
		@rm -f $(HISTO) $(HISTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(HISTDEP) $(HISTDS) $(HISTDH) $(HISTLIB) $(HISTMAP)
		@rm -rf include/v5

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(HISTDO): NOOPT = $(OPT)
