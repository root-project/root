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
ROOTCINTO    := $(ROOTCINTS:.cxx=.o)
ROOTCINT7O   := $(ROOTCINT7S:.cxx=.o)
ROOTCINTDEP  := $(ROOTCINTO:.o=.d) $(ROOTCINT7O:.o=.d) \
                $(ROOTCINTTMPO:.o=.d) $(ROOTCINT7TMPO:.o=.d) 

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
		   $(ROOTCINTTMPO) $(METAUTILSO) $(CINTTMPO) $(CINTTMPLIBS) $(CILIBS)

ifneq ($(BUILDBOTHCINT),)
.PRECIOUS: $(ROOTCINT7S) $(METAUTILS7S)

%7.cxx: %.cxx
		cp $< $@

$(ROOTCINT7EXE): $(CINT7LIB) $(ROOTCINT7O) $(METAUTILS7O) $(IOSENUM)
		$(LD) $(LDFLAGS) -o $@ $(ROOTCINT7O) $(METAUTILS7O) \
		   $(RPATH) $(CINT7LIBS) $(CILIBS)

$(ROOTCINT7TMPEXE): $(CINT7TMPO) $(ROOTCINT7TMPO) $(METAUTILS7O) \
                    $(IOSENUM) $(REFLEXO)
		$(LD) $(LDFLAGS) -o $@ \
		   $(ROOTCINT7TMPO) $(METAUTILS7O) $(CINT7TMPO) $(REFLEXO) $(CILIBS)
endif

$(RLIBMAP):     $(RLIBMAPO)
ifneq ($(PLATFORM),win32)
		$(LD) $(LDFLAGS) -o $@ $<
else
		$(LD) $(LDFLAGS) -o $@ $< imagehlp.lib
endif

all-$(MODNAME): $(ROOTCINTTMPEXE) $(ROOTCINTEXE) \
                $(ROOTCINT7TMPEXE) $(ROOTCINT7EXE) \
                $(RLIBMAP)

clean-$(MODNAME):
		@rm -f $(ROOTCINTTMPO) $(ROOTCINT7TMPO) $(ROOTCINTO) $(ROOTCINT7O) \
		  $(RLIBMAPO) 

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(ROOTCINTDEP) $(ROOTCINTTMPEXE) $(ROOTCINT7TMPEXE) \
		   $(ROOTCINTEXE) $(ROOTCINT7EXE) \
		   $(RLIBMAPDEP) $(RLIBMAP) \
		   $(UTILSDIRS)/*.exp $(UTILSDIRS)/*.lib $(UTILSDIRS)/*_tmp.cxx

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(UTILSDIRS)%_tmp.cxx: $(UTILSDIRS)%.cxx
	cp $< $@

$(ROOTCINTTMPO):  CXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD
$(ROOTCINTTMPO):  PCHCXXFLAGS =
$(ROOTCINTO):     PCHCXXFLAGS =
$(RLIBMAPO):      PCHCXXFLAGS =
$(ROOTCINT7TMPO): CXXFLAGS += -UR__HAVE_CONFIG -DROOTBUILD -DR__BUILDING_CINT7
$(ROOTCINT7TMPO): PCHCXXFLAGS =
$(ROOTCINT7O):    PCHCXXFLAGS =
$(ROOTCINT7O):    CXXFLAGS += -DR__BUILDING_CINT7
