# ModuleVars.mk for utils module
# Copyright (c) 2008 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2008-06-10

ifneq ($(HOST),)

UTILSDIRS    := $(BUILDTOOLSDIR)/core/utils/src

ROOTCINTS    := $(UTILSDIRS)/rootcint.cxx \
                $(filter-out %_tmp.cxx,$(wildcard $(UTILSDIRS)/R*.cxx))
ROOTCINTTMPO := $(ROOTCINTS:.cxx=_tmp.o)
ROOTCINTTMPEXE := $(UTILSDIRS)/rootcint_tmp$(EXEEXT)
ROOTCINTTMP  ?= $(ROOTCINTTMPEXE) -$(ROOTDICTTYPE)

ifeq ($(BUILDCLING),yes)
ROOTCLINGS    := $(UTILSDIRS)/rootcint.cxx \
                $(filter-out %_tmp.cxx,$(wildcard $(UTILSDIRS)/R*.cxx))
ROOTCLINGTMPO := $(ROOTCLINGS:.cxx=_tmp.o)
ROOTCLINGTMPEXE := $(UTILSDIRS)/rootcint_tmp$(EXEEXT)
ROOTCLINGTMP  ?= $(ROOTCLINGTMPEXE) -$(ROOTDICTTYPE)
endif

##### Dependencies for all dictionaries
ifeq ($(BUILDCLING),yes)
ROOTCINTTMPDEP = $(ROOTCLINGTMPO) $(ORDER_) $(ROOTCLINGTMPEXE)
else
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)
endif

##### rlibmap #####
RLIBMAP      := $(BUILDTOOLSDIR)/bin/rlibmap$(EXEEXT)

else

MODNAME      := utils
MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
UTILSDIR     := $(MODDIR)
UTILSDIRS    := $(UTILSDIR)/src
UTILSDIRI    := $(UTILSDIR)/inc

##### rootcint #####
ROOTCINTS    := $(UTILSDIRS)/rootcint.cxx \
                $(filter-out %_tmp.cxx,$(wildcard $(UTILSDIRS)/R*.cxx))
ROOTCINTTMPO := $(call stripsrc,$(ROOTCINTS:.cxx=_tmp.o))

ROOTCINTTMPEXE := $(call stripsrc,$(UTILSDIRS)/rootcint_tmp$(EXEEXT))
ROOTCINTEXE  := bin/rootcint$(EXEEXT)
ROOTCINTTMP  ?= $(ROOTCINTTMPEXE) -$(ROOTDICTTYPE)

##### rootcint #####
ifeq ($(BUILDCLING),yes)
ROOTCLINGS    := $(UTILSDIRS)/rootcint.cxx \
                $(filter-out %_tmp.cxx,$(wildcard $(UTILSDIRS)/R*.cxx))
ROOTCLINGTMPO := $(call stripsrc,$(ROOTCLINGS:.cxx=_tmp.o))

ROOTCLINGTMPEXE := $(call stripsrc,$(UTILSDIRS)/rootcint_tmp$(EXEEXT))
ROOTCLINGEXE  := bin/rootcint$(EXEEXT)
ROOTCLINGTMP  ?= $(ROOTCLINGTMPEXE) -$(ROOTDICTTYPE)
endif

##### Dependencies for all dictionaries
ifeq ($(BUILDCLING),yes)
ROOTCINTTMPDEP = $(ROOTCLINGTMPO) $(ORDER_) $(ROOTCLINGTMPEXE)
else
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)
endif

##### rlibmap #####
RLIBMAP      := bin/rlibmap$(EXEEXT)

endif
