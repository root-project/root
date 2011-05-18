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

##### Dependencies for all dictionaries
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)

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

##### Dependencies for all dictionaries
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)

##### rlibmap #####
RLIBMAP      := bin/rlibmap$(EXEEXT)

endif
