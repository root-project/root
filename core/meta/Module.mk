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

##### libMeta (part of libCore) #####
METAL        := $(MODDIRI)/LinkDef.h
METADS       := $(call stripsrc,$(MODDIRS)/G__Meta.cxx)
METADO       := $(METADS:.cxx=.o)
METADH       := $(METADS:.cxx=.h)

METAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
METAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))
METADCLINGCXXFLAGS:= -DR__WITH_CLING
METACLINGCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
METACLINGCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif
METAO        := $(call stripsrc,$(METAS:.cxx=.o))

METADEP      := $(METAO:.o=.d) $(METADO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAH))

# include all dependency files
INCLUDEFILES += $(METADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METADIRI)/%.h
		cp $< $@

$(METADS):      $(METAH) $(METAL) $(ROOTCINTTMPDEP) $(LLVMDEP)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCINTTMP) -f $@ -c -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS -Ietc -Ietc/cling $(METADCLINGCXXFLAGS) $(METAH) cling/Interpreter/Interpreter.h cling/Interpreter/DynamicExprInfo.h cling/Interpreter/DynamicLookupLifetimeHandler.h cling/Interpreter/StoredValueRef.h $(METAL)

all-$(MODNAME): $(METAO) $(METADO)

clean-$(MODNAME):
		@rm -f $(METAO) $(METADO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METADEP) $(METADS) $(METADH)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
$(METADO): NOOPT = $(OPT)
$(call stripsrc,$(patsubst %.cxx,%.o,$(wildcard $(MODDIRS)/TCling*.cxx))): \
   $(LLVMDEP)
$(call stripsrc,$(patsubst %.cxx,%.o,$(wildcard $(MODDIRS)/TInterpreter*.cxx))): \
   $(LLVMDEP)
$(call stripsrc,$(patsubst %.cxx,%.o,$(wildcard $(MODDIRS)/TCling*.cxx))): \
   CXXFLAGS += $(METACLINGCXXFLAGS)
$(call stripsrc,$(patsubst %.cxx,%.o,$(wildcard $(MODDIRS)/TInterpreter*.cxx))): \
   CXXFLAGS += $(METACLINGCXXFLAGS)
$(call stripsrc,$(MODDIRS)/TClingCallbacks.o): \
   CXXFLAGS += -fno-rtti

ifeq ($(ARCH),win32gcc)
# for EnumProcessModules():
CORELIBEXTRA += -lpsapi
endif

ifneq ($(CXX:g++=),$(CXX))
METADOCXXFLAGS := -Wno-shadow -Wno-unused-parameter
endif
$(METADO): CXXFLAGS += -D__CLING__ -Ietc -Ietc/cling $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS))) $(METADOCXXFLAGS)
