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
ROOTCINTTMPEXE:= $(UTILSDIRS)/rootcint_tmp$(EXEEXT)
ROOTCINTEXE  := bin/rootcint$(EXEEXT)
ROOTCINTTMP  ?= $(ROOTCINTTMPEXE) -$(ROOTDICTTYPE)
##### Dependencies for all dictionaries
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)

##### rlibmap #####
RLIBMAP      := bin/rlibmap$(EXEEXT)
