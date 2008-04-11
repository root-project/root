# Module.mk for tmva module
# Copyright (c) 2006 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 20/6/2005

MODNAME      := tmva
MODDIR       := $(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

TMVADIR      := $(MODDIR)
TMVADIRS     := $(TMVADIR)/src
TMVADIRI     := $(TMVADIR)/inc

##### libTMVA #####
TMVAL        := $(MODDIRI)/LinkDef.h
TMVADS       := $(MODDIRS)/G__TMVA.cxx
TMVADO       := $(TMVADS:.cxx=.o)
TMVADH       := $(TMVADS:.cxx=.h)

TMVAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
TMVAH_CINT   := $(subst tmva/inc,include/TMVA,$(TMVAH))
TMVAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
TMVAO        := $(TMVAS:.cxx=.o)

TMVADEP      := $(TMVAO:.o=.d) $(TMVADO:.o=.d)

TMVALIB      := $(LPATH)/libTMVA.$(SOEXT)
TMVAMAP      := $(TMVALIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/TMVA/%.h,$(TMVAH))
ALLLIBS      += $(TMVALIB)
ALLMAPS      += $(TMVAMAP)

# include all dependency files
INCLUDEFILES += $(TMVADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/TMVA/%.h: $(TMVADIRI)/%.h
		@(if [ ! -d "include/TMVA" ]; then     \
		   mkdir -p include/TMVA;              \
		fi)
		cp $< $@

$(TMVALIB):     $(TMVAO) $(TMVADO) $(ORDER_) $(MAINLIBS) $(TMVALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libTMVA.$(SOEXT) $@ "$(TMVAO) $(TMVADO)" \
		   "$(TMVALIBEXTRA)"

$(TMVADS):      $(TMVAH_CINT) $(TMVAL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(TMVAH_CINT) $(TMVAL)

$(TMVAMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(TMVAL)
		$(RLIBMAP) -o $(TMVAMAP) -l $(TMVALIB) \
		   -d $(TMVALIBDEPM) -c $(TMVAL)

all-$(MODNAME): $(TMVALIB) $(TMVAMAP)

clean-$(MODNAME):
		@rm -f $(TMVAO) $(TMVADO)

clean::         clean-tmva

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(TMVADEP) $(TMVADS) $(TMVADH) $(TMVALIB) $(TMVAMAP)
		@rm -rf include/TMVA

distclean::     distclean-$(MODNAME)
