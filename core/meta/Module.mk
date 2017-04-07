# Module.mk for meta module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := meta
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

METADIR      := $(MODDIR)
METADIRS     := $(METADIR)/src
METADIRI     := $(METADIR)/inc
METADIRR     := $(METADIR)/res

##### libMeta (part of libCore) #####
METAL        := $(MODDIRI)/LinkDef.h

METAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
METAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

# exclude this file from the dictionary
METASEL      := $(MODDIRI)/RootMetaSelection.h
METADICTH    := $(filter-out $(METASEL),$(METAH))
METAO        := $(call stripsrc,$(METAS:.cxx=.o))

METADEP      := $(METAO:.o=.d) $(METADO:.o=.d)

# used in the main Makefile
METAH_REL   := $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAH))
ALLHDRS     += $(METAH_REL)

# include all dependency files
INCLUDEFILES += $(METADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METADIRI)/%.h
		cp $< $@

all-$(MODNAME): $(METAO)
clean-$(MODNAME):
		@rm -f $(METAO) $(GLINGDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METADEP) $(CLINGDS) $(CLINGDH) $(CLINGLIB) $(CLINGMAP)

distclean::     distclean-$(MODNAME)

ifneq (,$(filter $(ARCH),win32gcc win64gcc))
# for EnumProcessModules():
CORELIBEXTRA += -lpsapi
endif

ifneq ($(CXX:g++=),$(CXX))
METADOCXXFLAGS := -Wno-shadow -Wno-unused-parameter
endif
# $(COREDO): CXXFLAGS += -D__CLING__ -Ietc -Ietc/cling $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS))) $(METADOCXXFLAGS)
$(METAO):  CXXFLAGS += -I$(FOUNDATIONDIRR) -I$(METADIRR)
