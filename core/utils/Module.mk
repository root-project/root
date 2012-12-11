# Module.mk for utils module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

# see also ModuleVars.mk

MODNAME := utils

ifneq ($(HOST),)

.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

all-$(MODNAME):

clean-$(MODNAME):

distclean-$(MODNAME):

else # ifneq ($(HOST),)

.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

.SECONDARY: $(ROOTCLINGTMPS)

$(ROOTCLINGEXE): $(ROOTCLINGO) $(ROOTCLINGUTILO) \
	   $(METAUTILSO) $(METAUTILSTO) $(SNPRINTFO) $(STRLCPYO) \
	   $(CLINGO)
	$(LD) $(LDFLAGS) -o $@ $(ROOTCLINGO) $(ROOTCLINGUTILO) \
	   $(METAUTILSO) $(METAUTILSTO) $(SNPRINTFO) $(STRLCPYO) \
	   $(CLINGO) $(CLINGLIBEXTRA) $(RPATH) $(CILIBS)

$(ROOTCLINGTMPEXE): $(CINTTMPO) $(ROOTCLINGTMPO) $(ROOTCLINGUTILO) \
	   $(METAUTILSO) $(METAUTILSTO) $(SNPRINTFO) $(STRLCPYO) \
	   $(CLINGO)
	$(LD) $(LDFLAGS) -o $@ $(ROOTCLINGTMPO) $(ROOTCLINGUTILO) \
	   $(METAUTILSO) $(METAUTILSTO) $(SNPRINTFO) $(STRLCPYO) \
	   $(CINTTMPLIBS) $(CLINGO) $(CLINGLIBEXTRA) $(CILIBS)

$(ROOTCINTEXE): $(ROOTCLINGEXE)
	cp -f $(ROOTCLINGEXE) $(ROOTCINTEXE)

ifneq ($(PLATFORM),win32)
$(RLIBMAP): $(RLIBMAPO)
	$(LD) $(LDFLAGS) -o $@ $<
else
$(RLIBMAP): $(RLIBMAPO)
	$(LD) $(LDFLAGS) -o $@ $< imagehlp.lib
endif

all-$(MODNAME): $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) $(ROOTCINTEXE) $(RLIBMAP)

clean-$(MODNAME):
	@rm -f $(ROOTCLINGTMPO) $(ROOTCLINGO) $(ROOTCLINGUTILO) $(RLIBMAPO)

clean:: clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -f $(ROOTCLINGDEP) $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) \
	   $(ROOTCINTEXE) $(RLIBMAPDEP) $(RLIBMAP) \
	   $(call stripsrc,$(UTILSDIRS)/*.exp $(UTILSDIRS)/*.lib \
	      $(UTILSDIRS)/*_tmp.cxx)

distclean:: distclean-$(MODNAME)

##### extra rules ######
$(call stripsrc,$(UTILSDIRS)/%_tmp.cxx): $(UTILSDIRS)/%.cxx
	$(MAKEDIR)
	cp $< $@

$(call stripsrc,$(UTILSDIRS)/rootcling_tmp.o): $(call stripsrc,\
	   $(UTILSDIRS)/rootcling_tmp.cxx)

$(call stripsrc,$(UTILSDIRS)/RStl_tmp.o): $(call stripsrc,\
	   $(UTILSDIRS)/RStl_tmp.cxx)

$(ROOTCLINGTMPO): $(LLVMDEP)
$(ROOTCLINGTMPO): CXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD -I$(UTILSDIRS) \
	   $(ROOTCLINGCXXFLAGS)
$(ROOTCLINGO): $(LLVMDEP)
$(ROOTCLINGO): CXXFLAGS += -UR__HAVE_CONFIG -I$(UTILSDIRS) $(ROOTCLINGCXXFLAGS)
$(ROOTCLINGUTILO): $(LLVMDEP)
$(ROOTCLINGUTILO): CXXFLAGS += -UR__HAVE_CONFIG -I$(UTILSDIRS) \
	   $(ROOTCLINGCXXFLAGS)

endif # ifneq ($(HOST),)
