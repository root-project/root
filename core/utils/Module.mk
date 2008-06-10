# Module.mk for utils module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := utils
MODDIR       := core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

# see also ModuleVars.mk

##### rootcint #####
ROOTCINTS    := $(MODDIRS)/rootcint.cxx \
                $(filter-out %_tmp.cxx,$(wildcard $(MODDIRS)/R*.cxx))
ROOTCINTO    := $(ROOTCINTS:.cxx=.o)
ROOTCINTTMPO := $(ROOTCINTS:.cxx=_tmp.o)
ROOTCINTDEP  := $(ROOTCINTO:.o=.d) $(ROOTCINTTMPO:.o=.d) 

##### rlibmap #####
RLIBMAPS     := $(MODDIRS)/rlibmap.cxx
RLIBMAPO     := $(RLIBMAPS:.cxx=.o)
RLIBMAPDEP   := $(RLIBMAPO:.o=.d)

# include all dependency files
INCLUDEFILES += $(ROOTCINTDEP) $(RLIBMAPDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(ROOTCINTEXE): $(CINTLIB) $(ROOTCINTO) $(METAUTILSO) $(IOSENUM)
		$(LD) $(LDFLAGS) -o $@ $(ROOTCINTO) $(METAUTILSO) \
		   $(RPATH) $(CINTLIBS) $(CILIBS)

$(ROOTCINTTMPEXE): $(CINTTMPO) $(ROOTCINTTMPO) $(METAUTILSO) $(IOSENUM)
		$(LD) $(LDFLAGS) -o $@ \
		   $(ROOTCINTTMPO) $(METAUTILSO) $(CINTTMPO) $(CILIBS)

$(RLIBMAP):     $(RLIBMAPO)
ifneq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) -o $@ $<
else
		$(LD) $(LDFLAGS) -o $@ $< imagehlp.lib
endif

all-$(MODNAME): $(ROOTCINTTMPEXE) $(ROOTCINTEXE) $(RLIBMAP)

clean-$(MODNAME):
		@rm -f $(ROOTCINTTMPO) $(ROOTCINTO) $(RLIBMAPO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTCINTDEP) $(ROOTCINTTMPEXE) $(ROOTCINTEXE) \
		   $(RLIBMAPDEP) $(RLIBMAP) \
		   $(UTILSDIRS)/*.exp $(UTILSDIRS)/*.lib $(UTILSDIRS)/*_tmp.cxx

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(UTILSDIRS)%_tmp.cxx: $(UTILSDIRS)%.cxx
	cp -f $< $@

$(ROOTCINTTMPO): CXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD
$(ROOTCINTTMPO): PCHCXXFLAGS =
$(ROOTCINTO):    PCHCXXFLAGS =
$(RLIBMAPO):     PCHCXXFLAGS =
