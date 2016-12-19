# ModuleVars.mk for main module
# Copyright (c) 1995-2016 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2016-12-19

##### rootcling #####
ROOTCLINGEXE    := bin/rootcling$(EXEEXT)
ROOTCLINGSTAGE2 := $(ROOTCLINGEXE) -rootbuild
# Dependencies for all dictionaries
ROOTCLINGSTAGE2DEP := $(ROOTCLINGEXE)
