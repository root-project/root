# Module.mk for base module
# Copyright (c) 2000 Rene Brun and Fons Rademakers
#
# Author: Fons Rademakers, 29/2/2000

MODNAME      := base
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

BASEDIR      := $(MODDIR)
BASEDIRS     := $(BASEDIR)/src
BASEDIRI     := $(BASEDIR)/inc

##### libBase (part of libCore) #####
BASEL1       := $(MODDIRI)/LinkDef1.h
BASEL2       := $(MODDIRI)/LinkDef2.h
BASEL3       := $(MODDIRI)/LinkDef3.h

BASEH1       := $(wildcard $(MODDIRI)/T*.h)
BASEH3       := GuiTypes.h KeySymbols.h Buttons.h TTimeStamp.h TVirtualMutex.h \
                TVirtualPerfStats.h TVirtualX.h TParameter.h \
                TVirtualAuth.h TFileInfo.h TFileCollection.h \
                TRedirectOutputGuard.h TVirtualMonitoring.h TObjectSpy.h \
                TUri.h TUrl.h TInetAddress.h TVirtualTableInterface.h \
                TBase64.h
BASEH3       := $(patsubst %,$(MODDIRI)/%,$(BASEH3))
BASEH1       := $(filter-out $(BASEH3),$(BASEH1))
BASEH        := $(filter-out $(MODDIRI)/LinkDef%,$(wildcard $(MODDIRI)/*.h))
BASEDICTH    := $(BASEH1) $(BASEH3)
ROOTAS       := $(MODDIRS)/roota.cxx
ROOTAO       := $(call stripsrc,$(ROOTAS:.cxx=.o))
BASES        := $(filter-out $(ROOTAS),\
		$(filter-out $(MODDIRS)/G__%,\
		$(wildcard $(MODDIRS)/*.cxx)))
BASEO        := $(call stripsrc,$(BASES:.cxx=.o))

BASEDEP      := $(BASEO:.o=.d) $(ROOTAO:.o=.d)

# used in the main Makefile
ALLHDRS     += $(patsubst $(MODDIRI)/%.h,include/%.h,$(BASEH))

# include all dependency files
INCLUDEFILES += $(BASEDEP)

##### local rules #####
.PHONY:         all-$(MODNAME) clean-$(MODNAME) distclean-$(MODNAME)

include/%.h:    $(BASEDIRI)/%.h
		cp $< $@

# Explicitely state this dependency.
# rmkdepend does not pick it up if $(COMPILEDATA) doesn't exist yet.
$(call stripsrc,$(BASEDIRS)/TSystem.d $(BASEDIRS)/TSystem.o): $(COMPILEDATA)
$(call stripsrc,$(BASEDIRS)/TROOT.d $(BASEDIRS)/TROOT.o): $(COMPILEDATA)

all-$(MODNAME): $(BASEO)

clean-$(MODNAME):
		@rm -f $(BASEO)

clean::         clean-$(MODNAME)

distclean-$(MODNAME): clean-$(MODNAME)
		@rm -f $(BASEDEP)

distclean::     distclean-$(MODNAME)

##### extra rules ######
$(call stripsrc,$(BASEDIRS)/TROOT.o): $(RGITCOMMITH)
# RGITCOMMITH depends on COREO *except* for TROOT.o (because of circular
# dependencies). But a rebuild of TROOT.o should trigger a rebuild of
# RGITCOMMITH, too - thus add these dependencies here.
$(RGITCOMMITH): $(BASEDIRS)/TROOT.cxx $(BASEDIRI)/TROOT.h

$(call stripsrc,$(BASEDIRS)/TPRegexp.o): $(PCREDEP)
$(call stripsrc,$(BASEDIRS)/TPRegexp.o): CXXFLAGS += $(PCREINC)

$(call stripsrc,$(BASEDIRS)/TROOT.o): CXXFLAGS += -Icore/base/src

ifeq ($(GCC_MAJOR),4)
ifeq ($(GCC_MINOR),1)
$(call stripsrc,$(BASEDIRS)/TString.o): CXXFLAGS += -Wno-strict-aliasing
$(call stripsrc,$(BASEDIRS)/TContextMenu.o): CXXFLAGS += -Wno-strict-aliasing
endif
endif

$(COREDO): $(PCREDEP)
$(COREDO): CXXFLAGS += $(PCREINC)
ifeq ($(ARCH),linuxicc)
$(COREDO):     CXXFLAGS += -wd191
endif

# rebuild after reconfigure
$(call stripsrc,$(BASEDIRS)/TROOT.o): config/Makefile.config
