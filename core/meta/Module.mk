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

METAH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
METAS        := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

METASLLVM    := $(MODDIRS)/TCling.cxx \
                $(MODDIRS)/TClingBaseClassInfo.cxx \
                $(MODDIRS)/TClingCallFunc.cxx \
                $(MODDIRS)/TClingCallbacks.cxx \
                $(MODDIRS)/TClingClassInfo.cxx \
                $(MODDIRS)/TClingDataMemberInfo.cxx \
                $(MODDIRS)/TClingMethodArgInfo.cxx \
                $(MODDIRS)/TClingMethodInfo.cxx \
                $(MODDIRS)/TClingTypeInfo.cxx \
                $(MODDIRS)/TClingTypedefInfo.cxx \
                $(MODDIRS)/TClingValue.cxx

METADCLINGCXXFLAGS:= -DR__WITH_CLING
METACLINGCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
METACLINGCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif
# exclude this file from the dictionary
METASEL      := $(MODDIRI)/RootMetaSelection.h
METADICTH    := $(filter-out $(METASEL),$(METAH))
METAO        := $(call stripsrc,$(METAS:.cxx=.o))

METADEP      := $(METAO:.o=.d) $(METADO:.o=.d)

METAOLLVM    := $(call stripsrc,$(METASLLVM:.cxx=.o))
METAO        := $(filter-out $(METAOLLVM),$(METAO))

##### libCling #####

CLINGL       := $(MODDIRI)/LinkDefCling.h
CLINGDS      := $(call stripsrc,$(MODDIRS)/G__Cling.cxx)
#CLINGDO      := $(CLINGDS:.cxx=.o)
CLINGDO      :=
CLINGDH      := $(CLINGDS:.cxx=.h)
CLINGH       := $(MODDIRS)/TCling.h

CLINGLIB     := $(LPATH)/libCling.$(SOEXT)
CLINGMAP     := $(CLINGLIB:.$(SOEXT)=.rootmap)

ALLLIBS      += $(CLINGLIB)
ALLMAPS      += $(CLINGMAP)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAH))

# include all dependency files
INCLUDEFILES += $(METADEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METADIRI)/%.h
		cp $< $@

IOLIB_EARLY = $(LPATH)/libRIO.$(SOEXT)

$(CLINGLIB):    $(CLINGO) $(CLINGDO) $(METAOLLVM) $(METAUTILSOLLVM) \
                $(METAUTILSTO) $(ORDER_) $(MAINLIBS) $(TCLINGLIBDEPM) $(IOLIB_EARLY)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libCling.$(SOEXT) $@ \
		   "$(METAOLLVM) $(METAUTILSOLLVM) $(METAUTILSTO) \
		    $(CLINGO) $(CLINGDO) $(CLINGLIBEXTRA) $(TCLINGLIBEXTRA)" \
		   ""

$(CLINGMAP):    $(CLINGL) $(ROOTCLINGSTAGE1DEP) $(LLVMDEP) $(call pcmdep,CLING)
		$(MAKEDIR)
		@echo "Generating rootmap $@..."
		$(ROOTCLINGSTAGE1) -r $(CLINGDS) $(call dictModule,CLING) -c \
		   $(CLINGH) $(CLINGL)

$(call pcmrule,CLING)
	$(noop)

$(CLINGDS): $(CLINGL) $(ROOTCLINGSTAGE1DEP) $(LLVMDEP) $(call pcmdep,CLING)
		$(MAKEDIR)
		@echo "Generating dictionary $@..."
		$(ROOTCLINGSTAGE1) -f $@ $(call dictModule,CLING) -c $(CLINGH) \
		   $(CLINGL)

all-$(MODNAME): $(METAO) $(METAOLLVM) $(CLINGLIB)
clean-$(MODNAME):
		@rm -f $(METAO) $(METAOLLVM) $(GLINGDO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METADEP) $(CLINGDS) $(CLINGDH) $(CLINGLIB) $(CLINGMAP)

distclean::     distclean-$(MODNAME)

# Optimize dictionary with stl containers.
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

ifneq (,$(filter $(ARCH),win32gcc win64gcc))
# for EnumProcessModules():
CORELIBEXTRA += -lpsapi
endif

ifneq ($(CXX:g++=),$(CXX))
METADOCXXFLAGS := -Wno-shadow -Wno-unused-parameter
endif
$(COREDO): CXXFLAGS += -D__CLING__ -Ietc -Ietc/cling $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS))) $(METADOCXXFLAGS)
