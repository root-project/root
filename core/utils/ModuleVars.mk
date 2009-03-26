# ModuleVars.mk for utils module
# Copyright (c) 2008 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2008-06-10

MODNAME      := utils
MODDIR       := core/$(MODNAME)
UTILSDIR     := $(MODDIR)
UTILSDIRS    := $(UTILSDIR)/src
UTILSDIRI    := $(UTILSDIR)/inc

##### rootcint #####
ROOTCINTS    := $(UTILSDIRS)/rootcint.cxx \
                $(filter-out %7.cxx,$(filter-out %_tmp.cxx,$(wildcard $(UTILSDIRS)/R*.cxx)))
ROOTCINTTMPO := $(ROOTCINTS:.cxx=_tmp.o)

ROOTCINTTMPEXE:= $(UTILSDIRS)/rootcint_tmp$(EXEEXT)
ROOTCINTEXE  := bin/rootcint$(EXEEXT)
ROOTCINTTMP  ?= $(ROOTCINTTMPEXE) -$(ROOTDICTTYPE)

##### Dependencies for all dictionaries
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)

##### rootcint7 #####
ifneq ($(findstring $(MAKECMDGOALS),distclean maintainer-clean),)
BUILDBOTHCINT := yes
endif
ifneq ($(BUILDBOTHCINT),)
ROOTCINT7S    := $(patsubst %.cxx,%7.cxx,$(ROOTCINTS))
ROOTCINT7TMPO := $(ROOTCINT7S:.cxx=_tmp.o)

ROOTCINT7TMPEXE:= $(UTILSDIRS)/rootcint7_tmp$(EXEEXT)
ROOTCINT7EXE  := bin/rootcint7$(EXEEXT)
ROOTCINT7TMP  ?= $(ROOTCINT7TMPEXE) -$(ROOTDICTTYPE)

ROOTCINT7TMPDEP = $(ROOTCINT7TMPO) $(ORDER_) $(ROOTCINT7TMPEXE)
else
ROOTCINT7TMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)
endif

##### rlibmap #####
RLIBMAP      := bin/rlibmap$(EXEEXT)
