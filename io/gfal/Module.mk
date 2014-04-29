# Module.mk for gfal module
# Copyright (c) 2005 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 9/12/2005

MODNAME      := gfal
MODDIR       := $(ROOT_SRCDIR)/io/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

GFALDIR      := $(MODDIR)
GFALDIRS     := $(GFALDIR)/src
GFALDIRI     := $(GFALDIR)/inc

##### libGFAL #####
GFALL        := $(MODDIRI)/LinkDef.h
GFALDS       := $(call stripsrc,$(MODDIRS)/G__GFAL.cxx)
GFALDO       := $(GFALDS:.cxx=.o)
GFALDH       := $(GFALDS:.cxx=.h)

GFALH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
GFALS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
GFALO        := $(call stripsrc,$(GFALS:.cxx=.o))

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

$(call pcmrule,GFAL)
	$(noop)

$(GFALDS):      $(GFALH) $(GFALL) $(ROOTCLINGEXE) $(call pcmdep,GFAL)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE2) -f $@ $(call dictModule,GFAL) -c $(GFALH) $(GFALL)

$(GFALMAP):     $(GFALH) $(GFALL) $(ROOTCLINGEXE) $(call pcmdep,GFAL)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE2) -r $(GFALDS) $(call dictModule,GFAL) -c $(GFALH) $(GFALL)

all-$(MODNAME): $(GFALLIB)

clean-$(MODNAME):
		@rm -f $(GFALO) $(GFALDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(GFALDEP) $(GFALDS) $(GFALDH) $(GFALLIB) $(GFALMAP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(GFALO) $(GFALDO): CXXFLAGS := $(filter-out -Wshadow,$(CXXFLAGS))
$(GFALO): CXXFLAGS += $(GFALINCDIR:%=-I%)
$(GFALO): CXXFLAGS += $(SRMIFCEINCDIR:%=-I%)
$(GFALO): CXXFLAGS += $(GLIB2INCDIR:%=-I%)
$(GFALO): CXXFLAGS += -D_FILE_OFFSET_BITS=64
