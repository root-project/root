# Module.mk for splot module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 27/8/2003

MODNAME     := splot
MODDIR      := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS     := $(MODDIR)/src
MODDIRI     := $(MODDIR)/inc

SPLOTDIR    := $(MODDIR)
SPLOTDIRS   := $(SPLOTDIR)/src
SPLOTDIRI   := $(SPLOTDIR)/inc

##### libSPlot #####
SPLOTL      := $(MODDIRI)/LinkDef.h
SPLOTDS     := $(call stripsrc,$(MODDIRS)/G__SPlot.cxx)
SPLOTDO     := $(SPLOTDS:.cxx=.o)
SPLOTDH     := $(SPLOTDS:.cxx=.h)

SPLOTH      := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
SPLOTS      := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
SPLOTO      := $(call stripsrc,$(SPLOTS:.cxx=.o))

SPLOTDEP    := $(SPLOTO:.o=.d) $(SPLOTDO:.o=.d)

SPLOTLIB    := $(LPATH)/libSPlot.$(SOEXT)
SPLOTMAP    := $(SPLOTLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(SPLOTH))
ALLLIBS     += $(SPLOTLIB)
ALLMAPS     += $(SPLOTMAP)

# include all dependency files
INCLUDEFILES += $(SPLOTDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(SPLOTDIRI)/%.h
		cp $< $@

$(SPLOTLIB):    $(SPLOTO) $(SPLOTDO) $(ORDER_) $(MAINLIBS) $(SPLOTLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libSPlot.$(SOEXT) $@ "$(SPLOTO) $(SPLOTDO)" \
		   "$(SPLOTLIBEXTRA)"

$(call pcmrule,SPLOT)
	$(noop)

$(SPLOTDS):     $(SPLOTH) $(SPLOTL) $(ROOTCLINGEXE) $(call pcmdep,SPLOT)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,SPLOT) -c $(SPLOTH) $(SPLOTL)

$(SPLOTMAP):    $(SPLOTH) $(SPLOTL) $(ROOTCLINGEXE) $(call pcmdep,SPLOT)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(SPLOTDS) $(call dictModule,SPLOT) -c $(SPLOTH) $(SPLOTL)

all-$(MODNAME): $(SPLOTLIB)

clean-$(MODNAME):
		@rm -f $(SPLOTO) $(SPLOTDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(SPLOTDEP) $(SPLOTDS) $(SPLOTDH) $(SPLOTLIB) $(SPLOTMAP)

distclean::     distclean-$(MODNAME)
