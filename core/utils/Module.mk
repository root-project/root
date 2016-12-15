# Module.mk for utils module
# Copyright (c) 2016 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

# see also ModuleVars.mk

MODNAME      := utils
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UTILSDIR      := $(MODDIR)
UTILSDIRS     := $(UTILSDIR)/src
UTILSDIRI     := $(UTILSDIR)/inc
UTILSDIRR     := $(UTILSDIR)/res

##### $(UTILSO) (part of rootcling stage 2) #####
UTILSS        := $(MODDIRS)/rootclingIO.cxx
UTILSO        := $(call stripsrc,$(UTILSS:.cxx=.o))
UTILSDEP      := $(UTILSO:.o=.d) $(UTILSDO:.o=.d)

##### local rules #####

ifneq ($(HOST),)
.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

all-$(MODNAME):

clean-$(MODNAME):

distclean-$(MODNAME):

else # ifneq ($(HOST),)

.PHONY: all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

.SECONDARY: $(ROOTCLINGTMPS)

# The dependency on $(CLINGLIB) was added to prevent $(CLINGLIB) and
# $(ROOTCLINGEXE) from being linked in parallel.
$(ROOTCLINGEXE): $(ROOTCLINGO) $(UTILSO) $(ROOTCLINGLIBSDEP)
	$(LD) $(LDFLAGS) $(OSTHREADLIBDIR) $(OSTHREADLIB) -o $@ $(ROOTCLINGO) $(UTILSO) \
	   $(RPATH) $(ROOTCLINGLIBS) $(CILIBS) $(CORELIBEXTRA) \
	   $(PCRELDFLAGS) $(PCRELIB) $(CRYPTLIBS)

$(ROOTCLINGTMPEXE): $(ROOTCLINGTMPO) $(FOUNDATIONO) $(DICTGENO) $(CLINGUTILSO) \
	   $(SNPRINTFO) $(STRLCPYO) $(CLINGO)
	$(LD) $(LDFLAGS) $(OSTHREADLIBDIR) $(OSTHREADLIB) -o $@ \
	   $(ROOTCLINGTMPO) $(FOUNDATIONO) $(DICTGENO) $(CLINGUTILSO) \
       $(SNPRINTFO) $(STRLCPYO) $(CLINGO) $(CLINGLIBEXTRA) $(CILIBS)

$(ROOTCINTEXE): $(ROOTCLINGEXE)
	ln -f $(ROOTCLINGEXE) $(ROOTCINTEXE)

$(GENREFLEXEXE): $(ROOTCLINGEXE)
	ln -f $(ROOTCLINGEXE) $(GENREFLEXEXE)

all-$(MODNAME): $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) $(ROOTCINTEXE) \
                $(GENREFLEXEXE) $(UTILSO)

clean-$(MODNAME):
	@rm -f $(ROOTCLINGTMPO) $(ROOTCLINGO) $(UTILSO)

clean:: clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
	@rm -f $(ROOTCLINGDEP) $(ROOTCLINGTMPEXE) $(ROOTCLINGEXE) \
	   $(ROOTCINTEXE) $(GENREFLEXEXE) \
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
$(ROOTCLINGTMPO): CXXFLAGS += -UR__HAVE_CONFIG -DROOT_STAGE1_BUILD -I$(CLINGUTILSDIRR) -I$(DICTGENDIRR) \
	   $(ROOTCLINGCXXFLAGS)
$(ROOTCLINGO): $(LLVMDEP)
$(ROOTCLINGO): CXXFLAGS += -UR__HAVE_CONFIG -I$(CLINGUTILSDIRR) -I$(DICTGENDIRR) -I$(METADIRR) $(ROOTCLINGCXXFLAGS)

# the -rdynamic flag is needed on cygwin to make symbols visible to dlsym
ifneq (,$(filter $(ARCH),win32gcc win64gcc))
$(ROOTCLINGEXE): LDFLAGS += -rdynamic
endif

endif # ifneq ($(HOST),)
