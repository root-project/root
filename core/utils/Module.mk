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

CLINGMETAUTILSO := $(METAUTILSO) $(METAUTILSTO) $(METAUTILSOLLVM)

$(ROOTCLINGEXE): $(ROOTCLINGO) $(ROOTCLINGUTILO) \
	   $(CLINGMETAUTILSO) $(SNPRINTFO) $(STRLCPYO) $(CLINGO)
	$(LD) $(LDFLAGS) -o $@ $(ROOTCLINGO) $(ROOTCLINGUTILO) \
	   $(CLINGMETAUTILSO) $(SNPRINTFO) $(STRLCPYO) \
	   $(CLINGO) $(CLINGLIBEXTRA) $(RPATH) $(CILIBS)

$(ROOTCLINGTMPEXE): $(CINTTMPO) $(ROOTCLINGTMPO) $(ROOTCLINGUTILO) \
	   $(CLINGMETAUTILSO) $(SNPRINTFO) $(STRLCPYO) $(CLINGO)
	$(LD) $(LDFLAGS) -o $@ $(ROOTCLINGTMPO) $(ROOTCLINGUTILO) \
	   $(CLINGMETAUTILSO) $(SNPRINTFO) $(STRLCPYO) \
	   $(CINTTMPLIBS) $(CLINGO) $(CLINGLIBEXTRA) $(CILIBS)

$(ROOTCINTEXE): $(ROOTCLINGEXE)
	ln -sf `basename $(ROOTCLINGEXE)` $(ROOTCINTEXE)

$(GENREFLEXEXE): $(ROOTCLINGEXE)
	ln -sf `basename $(ROOTCLINGEXE)` $(GENREFLEXEXE)

ifneq ($(PLATFORM),win32)
$(RLIBMAP): $(RLIBMAPO)
	$(LD) $(LDFLAGS) -o $@ $<
else
$(RLIBMAP): $(RLIBMAPO)
	$(LD) $(LDFLAGS) -o $@ $< imagehlp.lib
endif

all-$(MODNAME): $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) $(ROOTCINTEXE) \
                $(GENREFLEXEXE) $(RLIBMAP)

clean-$(MODNAME):
	@rm -f $(ROOTCLINGTMPO) $(ROOTCLINGO) $(ROOTCLINGUTILO) $(RLIBMAPO)

clean:: clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -f $(ROOTCLINGDEP) $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) \
	   $(ROOTCINTEXE) $(GENREFLEXEXE) $(RLIBMAPDEP) $(RLIBMAP) \
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
$(ROOTCLINGTMPO): CXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD -I$(UTILSDIRS) -I$(METAUTILSDIRS) \
	   $(ROOTCLINGCXXFLAGS)
$(ROOTCLINGO): $(LLVMDEP)
$(ROOTCLINGO): CXXFLAGS += -UR__HAVE_CONFIG -I$(UTILSDIRS) -I$(METAUTILSDIRS) $(ROOTCLINGCXXFLAGS)
$(ROOTCLINGUTILO): $(LLVMDEP)
$(ROOTCLINGUTILO): CXXFLAGS += -UR__HAVE_CONFIG -I$(UTILSDIRS) -I$(METAUTILSDIRS) \
	   $(ROOTCLINGCXXFLAGS)

endif # ifneq ($(HOST),)
