# Module.mk for utilities for libMeta and rootcint
# Copyright (c) 1995-2016 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2016-12-14

MODNAME        := metacling
MODDIR         := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS        := $(MODDIR)/src
MODDIRI        := $(MODDIR)/inc

METACLINGDIR   := $(MODDIR)
METACLINGDIRS  := $(METACLINGDIR)/src
METACLINGDIRI  := $(METACLINGDIR)/inc
METACLINGDIRR  := $(METACLINGDIR)/res

##### $(METACLINGO) #####
METACLINGH     := $(METACLINGDIRS)/TCling.h
METACLINGS     := $(filter-out $(MODDIRS)/G__%,$(wildcard $(MODDIRS)/*.cxx))

METACLINGCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -fno-rtti,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
METACLINGCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif

METACLINGO     := $(call stripsrc,$(METACLINGS:.cxx=.o))

# METACLINGL     := $(MODDIRI)/LinkDef.h

METACLINGDEP   := $(METACLINGO:.o=.d)

# used in the main Makefile
# ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METACLINGH))

# include all dependency files
INCLUDEFILES += $(METACLINGDEP)


##### libCling #####

CLINGLIB     := $(LPATH)/libCling.$(SOEXT)
CLINGMAP     := $(CLINGLIB:.$(SOEXT)=.rootmap)

IOLIB_EARLY = $(LPATH)/libRIO.$(SOEXT)

$(CLINGLIB):    $(CLINGUTILSO) $(DICTGENO) $(METACLINGO) $(CLINGO) \
                $(ORDER_) $(MAINLIBS) $(TCLINGLIBDEPM) $(IOLIB_EARLY)
		@$(MAKELIB) $(PLATFORM) $(LD) "$(LDFLAGS)" \
		   "$(SOFLAGS)" libCling.$(SOEXT) $@ \
		   "$(CLINGUTILSO) $(DICTGENO) $(METACLINGO) $(CLINGO) \
		    $(CLINGLIBEXTRA) $(TCLINGLIBEXTRA)" \
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



ALLLIBS      += $(CLINGLIB)
ALLMAPS      += $(CLINGMAP)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(METAH))

# include all dependency files
INCLUDEFILES += $(METADEP)


##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(METACLINGDIRI)/%.h
		cp $< $@

all-$(MODNAME): $(METACLINGO) $(STLDICTS)  $(CLINGLIB)

clean-$(MODNAME):
		@rm -f $(METACLINGO) $(STLDICTS_OBJ) \
		   $(STLDICTS_DEP)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(METACLINGDEP) \
		   $(STLDICTS_OBJ) $(STLDICTS_DEP) $(STLDICTS_SRC) \
		   $(STLDICTS_HDR) $(STLDICTSMAPS)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(METACLINGO): CXXFLAGS += $(METACLINGCXXFLAGS) -I$(METACLINGDIRR) -I$(CLINGUTILSDIRR) -I$(FOUNDATIONDIRR)
$(METACLINGO): $(LLVMDEP)

# TClingCallbacks.o needs -fno-rtti
$(call stripsrc,$(MODDIRS)/TClingCallbacks.o): CXXFLAGS += -fno-rtti
