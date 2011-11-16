# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 2002 Rene Brun and Fons Rademakers
#
# Author: Philippe Canal 9/1/2004

MODNAME        := metautils
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

METAUTILSDIR   := $(MODDIR)
METAUTILSDIRS  := $(METAUTILSDIR)/src
METAUTILSDIRI  := $(METAUTILSDIR)/inc

##### $(METAUTILSO) #####
METAUTILSH     := $(filter-out $(MODDIRI)/TMetaUtils.%,\
  $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h)))
METAUTILSS     := $(filter-out $(MODDIRS)/TMetaUtils.%,\
  $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx)))

ifeq ($(BUILDCLING),yes)
METAUTILSTH     += $(MODDIRI)/TMetaUtils.h
METAUTILSTS     += $(MODDIRS)/TMetaUtils.cxx
METAUTILSCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
METAUTILSTCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif
endif

METAUTILSO     := $(call stripsrc,$(METAUTILSS:.cxx=.o))
METAUTILSTO    := $(call stripsrc,$(METAUTILSTS:.cxx=.o))

METAUTILSL     := $(MODDIRI)/LinkDef.h
METAUTILSDS    := $(call stripsrc,$(MODDIRS)/G__MetaUtils.cxx)
METAUTILSDO    := $(METAUTILSDS:.cxx=.o)
METAUTILSDH    := $(METAUTILSDS:.cxx=.h)

METAUTILSDEP   := $(METAUTILSO:.o=.d) $(METAUTILSDO:.o=.d) $(METAUTILSTO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAUTILSH) $(METAUTILSTH))

# include all dependency files
INCLUDEFILES += $(METAUTILSDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METAUTILSDIRI)/%.h
		cp $< $@

$(METAUTILSDS): $(METAUTILSH) $(METAUTILSL) $(ROOTCINTTMPDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -DG__API $(METAUTILSH) $(METAUTILSL)

all-$(MODNAME): $(METAUTILSO) $(METAUTILSDO)

clean-$(MODNAME):
		@rm -f $(METAUTILSO) $(METAUTILSDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METAUTILSDEP) $(METAUTILSDS) $(METAUTILSDH)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(METAUTILSO): CXXFLAGS += $(METAUTILSCXXFLAGS)
