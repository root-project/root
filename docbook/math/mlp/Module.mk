# Module.mk for mlp module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Rene Brun, 27/8/2003

MODNAME      := mlp
MODDIR       := $(ROOT_SRCDIR)/math/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MLPDIR       := $(MODDIR)
MLPDIRS      := $(MLPDIR)/src
MLPDIRI      := $(MLPDIR)/inc

##### libMLP #####
MLPL         := $(MODDIRI)/LinkDef.h
MLPDS        := $(call stripsrc,$(MODDIRS)/G__MLP.cxx)
MLPDO        := $(MLPDS:.cxx=.o)
MLPDH        := $(MLPDS:.cxx=.h)

MLPH         := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MLPS         := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MLPO         := $(call stripsrc,$(MLPS:.cxx=.o))

MLPDEP       := $(MLPO:.o=.d) $(MLPDO:.o=.d)

MLPLIB       := $(LPATH)/libMLP.$(SOEXT)
MLPMAP       := $(MLPLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MLPH))
ALLLIBS     += $(MLPLIB)
ALLMAPS     += $(MLPMAP)

# include all dependency files
INCLUDEFILES += $(MLPDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MLPDIRI)/%.h
		cp $< $@

$(MLPLIB):      $(MLPO) $(MLPDO) $(ORDER_) $(MAINLIBS) $(MLPLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMLP.$(SOEXT) $@ "$(MLPO) $(MLPDO)" \
		   "$(MLPLIBEXTRA)"

$(MLPDS):       $(MLPH) $(MLPL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MLPH) $(MLPL)

$(MLPMAP):      $(RLIBMAP) $(MAKEFILEDEP) $(MLPL)
		$(RLIBMAP) -o $@ -l $(MLPLIB) -d $(MLPLIBDEPM) -c $(MLPL)

all-$(MODNAME): $(MLPLIB) $(MLPMAP)

clean-$(MODNAME):
		@rm -f $(MLPO) $(MLPDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME):  clean-$(MODNAME)
		@rm -f $(MLPDEP) $(MLPDS) $(MLPDH) $(MLPLIB) $(MLPMAP)

distclean::     distclean-$(MODNAME)
