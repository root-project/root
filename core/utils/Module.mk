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

##### rootcling #####
ROOTCLINGO   := $(call stripsrc,$(ROOTCLINGS:.cxx=.o))
ROOTCLINGDEP := $(ROOTCINTO:.o=.d) $(ROOTCLINGTMPO:.o=.d)

##### rlibmap #####
RLIBMAPS     := $(MODDIRS)/rlibmap.cxx
RLIBMAPO     := $(call stripsrc,$(RLIBMAPS:.cxx=.o))
RLIBMAPDEP   := $(RLIBMAPO:.o=.d)

# include all dependency files
INCLUDEFILES += $(ROOTCINTDEP) $(ROOTCLINGDEP) $(RLIBMAPDEP)

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

$(ROOTCLINGEXE): $(CINTLIB) $(ROOTCLINGO) $(METAUTILSO) $(SNPRINTFO) \
                $(STRLCPYO) $(IOSENUM) $(CLINGDEP)
		$(LD) $(LDFLAGS) -o $@ $(ROOTCLINGO) $(METAUTILSO) \
		  $(SNPRINTFO) $(STRLCPYO) $(RPATH) -Llib -lCling $(CINTLIBS) $(CILIBS)

$(ROOTCLINGTMPEXE): $(CINTTMPO) $(ROOTCLINGTMPO) $(METAUTILSO) $(SNPRINTFO) \
                   $(STRLCPYO) $(IOSENUM) $(CLINGDEP)
		$(LD) $(LDFLAGS) -o $@ \
		  $(ROOTCLINGTMPO) $(METAUTILSO) $(SNPRINTFO) $(STRLCPYO) \
		  $(CINTTMPO) $(CINTTMPLIBS) -Llib -lCling $(CINTLIBS) $(CILIBS)

$(RLIBMAP):     $(RLIBMAPO)
ifneq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) -o $@ $<
else
		$(LD) $(LDFLAGS) -o $@ $< imagehlp.lib
endif

all-$(MODNAME): $(ROOTCINTTMPEXE) $(ROOTCINTEXE) $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) $(RLIBMAP)

clean-$(MODNAME):
		@rm -f $(ROOTCINTTMPO) $(ROOTCINTO) $(ROOTCLINGTMPO) $(ROOTCLINGO) $(RLIBMAPO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTCINTDEP) $(ROOTCINTTMPEXE) $(ROOTCINTEXE) \
		   $(ROOTCLINGDEP) $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) \
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
$(ROOTCLINGTMPO): CXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD -I$(UTILSDIRS) $(ROOTCLINGCXXFLAGS)
$(ROOTCLINGO):    CXXFLAGS += -UR__HAVE_CONFIG -I$(UTILSDIRS) $(ROOTCLINGCXXFLAGS)

endif
