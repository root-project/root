# Module.mk for monalisa module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Andreas Peters, 07/12/2005

MODNAME      := monalisa
MODDIR       := $(ROOT_SRCDIR)/net/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

MONALISADIR  := $(MODDIR)
MONALISADIRS := $(MONALISADIR)/src
MONALISADIRI := $(MONALISADIR)/inc

##### libMonaLisa #####
MONALISAL    := $(MODDIRI)/LinkDef.h
MONALISADS   := $(call stripsrc,$(MODDIRS)/G__MonaLisa.cxx)
MONALISADO   := $(MONALISADS:.cxx=.o)
MONALISADH   := $(MONALISADS:.cxx=.h)

MONALISAH    := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
MONALISAS    := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
MONALISAO    := $(call stripsrc,$(MONALISAS:.cxx=.o))

MONALISADEP  := $(MONALISAO:.o=.d) $(MONALISADO:.o=.d)

MONALISALIB  := $(LPATH)/libMonaLisa.$(SOEXT)
MONALISAMAP  := $(MONALISALIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS      += $(patsubst $(MODDIRI)/%.h,include/%.h,$(MONALISAH))
ALLLIBS      += $(MONALISALIB)
ALLMAPS      += $(MONALISAMAP)

# include all dependency files
INCLUDEFILES += $(MONALISADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(MONALISADIRI)/%.h
		cp $< $@

$(MONALISALIB): $(MONALISAO) $(MONALISADO) $(ORDER_) $(MAINLIBS) $(MONALISALIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libMonaLisa.$(SOEXT) $@ "$(MONALISAO) $(MONALISADO)" \
		   "$(MONALISALIBEXTRA) $(MONALISALIBDIR) $(MONALISACLILIB)"

$(MONALISADS):  $(MONALISAH) $(MONALISAL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(MONALISAINCDIR:%=-I%) $(MONALISAH) $(MONALISAL)

$(MONALISAMAP): $(RLIBMAP) $(MAKEFILEDEP) $(MONALISAL)
		$(RLIBMAP) -o $@ -l $(MONALISALIB) \
		   -d $(MONALISALIBDEPM) -c $(MONALISAL)

all-$(MODNAME): $(MONALISALIB) $(MONALISAMAP)

clean-$(MODNAME):
		@rm -f $(MONALISAO) $(MONALISADO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(MONALISADEP) $(MONALISADS) $(MONALISADH) \
		   $(MONALISALIB) $(MONALISAMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(MONALISAO) $(MONALISADO): CXXFLAGS += $(MONALISAINCDIR:%=-I%)
