# Module.mk for gfal module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 9/12/2005

MODNAME      := gfal
MODDIR       := io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GFALDIR      := $(MODDIR)
GFALDIRS     := $(GFALDIR)/src
GFALDIRI     := $(GFALDIR)/inc

##### libGFAL #####
GFALL        := $(MODDIRI)/LinkDef.h
GFALDS       := $(MODDIRS)/G__GFAL.cxx
GFALDO       := $(GFALDS:.cxx=.o)
GFALDH       := $(GFALDS:.cxx=.h)

GFALH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GFALS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GFALO        := $(GFALS:.cxx=.o)

GFALDEP      := $(GFALO:.o=.d) $(GFALDO:.o=.d)

GFALLIB      := $(LPATH)/libGFAL.$(SOEXT)
GFALMAP      := $(GFALLIB:.$(SOEXT)=.rootmap)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(GFALH))
ALLLIBS     += $(GFALLIB)
ALLMAPS     += $(GFALMAP)

# include all dependency files
INCLUDEFILES += $(GFALDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(GFALDIRI)/%.h
		cp $< $@

$(GFALLIB):     $(GFALO) $(GFALDO) $(ORDER_) $(MAINLIBS) $(GFALLIBDEP)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libGFAL.$(SOEXT) $@ "$(GFALO) $(GFALDO)" \
		   "$(GFALLIBEXTRA) $(GFALLIBDIR) $(GFALCLILIB)"

$(GFALDS):      $(GFALH) $(GFALL) $(ROOTCINTTMPDEP)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c $(GFALH) $(GFALL)

$(GFALMAP):     $(RLIBMAP) $(MAKEFILEDEP) $(GFALL)
		$(RLIBMAP) -o $(GFALMAP) -l $(GFALLIB) \
		   -d $(GFALLIBDEPM) -c $(GFALL)

all-$(MODNAME): $(GFALLIB) $(GFALMAP)

clean-$(MODNAME):
		@rm -f $(GFALO) $(GFALDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GFALDEP) $(GFALDS) $(GFALDH) $(GFALLIB) $(GFALMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GFALO): CXXFLAGS += $(GFALINCDIR:%=-I%)
