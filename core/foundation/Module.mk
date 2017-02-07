# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 1995-2016 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2016-12-14

MODNAME        := foundation
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

FOUNDATIONDIR   := $(MODDIR)
FOUNDATIONDIRS  := $(FOUNDATIONDIR)/src
FOUNDATIONDIRI  := $(FOUNDATIONDIR)/inc
FOUNDATIONDIRR  := $(FOUNDATIONDIR)/res

##### $(FOUNDATIONO) #####
FOUNDATIONH     := $(filter-out $(MODDIRI)/libcpp_string_view.h,\
  $(filter-out $(MODDIRI)/RWrap_libcpp_string_view.h,\
  $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))))
FOUNDATIONS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

FOUNDATIONTH     += $(MODDIRI)/root_std_complex.h
FOUNDATIONTH     += $(MODDIRI)/libcpp_string_view.h
FOUNDATIONTH     += $(MODDIRI)/RWrap_libcpp_string_view.h
FOUNDATIONTH     += $(MODDIRI)/ROOT/RArrayView.hxx
FOUNDATIONTH     += $(MODDIRI)/ROOT/rhysd_array_view.hxx

FOUNDATIONO     := $(call stripsrc,$(FOUNDATIONS:.cxx=.o))

FOUNDATIONL     := $(MODDIRI)/LinkDef.h

FOUNDATIONDEP   := $(FOUNDATIONO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%,include/%,$(FOUNDATIONH) $(FOUNDATIONTH))

# include all dependency files
INCLUDEFILES += $(FOUNDATIONDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(FOUNDATIONDIRI)/%.h
		cp $< $@

include/%.hxx:	$(FOUNDATIONDIRI)/%.hxx
		mkdir -p include/ROOT
		cp $< $@

all-$(MODNAME): $(FOUNDATIONO)

clean-$(MODNAME):
		@rm -f $(FOUNDATIONO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(FOUNDATIONDEP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(FOUNDATIONO): CXXFLAGS += -I$(FOUNDATIONDIRR)
