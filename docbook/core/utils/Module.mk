# Module.mk for utils module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := utils

ifneq ($(HOST),)

.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

all-$(MODNAME):

clean-$(MODNAME):

distclean-$(MODNAME):

else

MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

# see also ModuleVars.mk

##### rootcint #####
ROOTCINTO    := $(call stripsrc,$(ROOTCINTS:.cxx=.o))
ROOTCINTDEP  := $(ROOTCINTO:.o=.d) $(ROOTCINTTMPO:.o=.d)

##### rlibmap #####
RLIBMAPS     := $(MODDIRS)/rlibmap.cxx
RLIBMAPO     := $(call stripsrc,$(RLIBMAPS:.cxx=.o))
RLIBMAPDEP   := $(RLIBMAPO:.o=.d)

# include all dependency files
INCLUDEFILES += $(ROOTCINTDEP) $(RLIBMAPDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

$(ROOTCINTEXE): $(CINTLIB) $(ROOTCINTO) $(METAUTILSO) $(SNPRINTFO) \
                $(STRLCPYO) $(IOSENUM)
		$(LD) $(LDFLAGS) -o $@ $(ROOTCINTO) $(METAUTILSO) \
		   $(SNPRINTFO) $(STRLCPYO) $(RPATH) $(CINTLIBS) $(CILIBS)

$(ROOTCINTTMPEXE): $(CINTTMPO) $(ROOTCINTTMPO) $(METAUTILSO) $(SNPRINTFO) \
                   $(STRLCPYO) $(IOSENUM)
		$(LD) $(LDFLAGS) -o $@ \
		   $(ROOTCINTTMPO) $(METAUTILSO) $(SNPRINTFO) $(STRLCPYO) \
		   $(CINTTMPO) $(CINTTMPLIBS) $(CILIBS)

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
		   $(call stripsrc,$(UTILSDIRS)/*.exp $(UTILSDIRS)/*.lib $(UTILSDIRS)/*_tmp.cxx)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(call stripsrc,$(UTILSDIRS)/%_tmp.cxx): $(UTILSDIRS)/%.cxx
	$(MAKEDIR)
	cp $< $@

$(call stripsrc,$(UTILSDIRS)/rootcint_tmp.o): $(call stripsrc,$(UTILSDIRS)/rootcint_tmp.cxx)

$(call stripsrc,$(UTILSDIRS)/RStl_tmp.o): $(call stripsrc,$(UTILSDIRS)/RStl_tmp.cxx)

$(ROOTCINTTMPO):  CXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD -I$(UTILSDIRS)

endif
