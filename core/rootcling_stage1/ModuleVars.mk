# ModuleVars.mk for rootcling_stage1 module
# Copyright (c) 2008 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2008-06-10

ifneq ($(HOST),)

ROOTCLING1DIRS  := $(BUILDTOOLSDIR)/core/rootcling_stage1/src

ROOTCLING1S     := $(ROOTCLING1DIRS)/rootcling_stage1.cxx
ROOTCLING1O     := $(ROOTCLING1S:.cxx=.o)
ROOTCLING1EXE   := $(UTILSDIRS)/rootcling_stage1$(EXEEXT)
ROOTCLINGSTAGE1 := $(ROOTCLING1EXE)

##### Dependencies for all dictionaries
ROOTCINT1DEP   = $(ROOTCLING1O) $(ROOTCLING1EXE)

else # ifneq ($(HOST),)

MODNAME      := rootcling_stage1

MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UTILSDIR     := $(MODDIR)
UTILSDIRS    := $(UTILSDIR)/src
UTILSDIRI    := $(UTILSDIR)/inc

##### rootcling #####
ROOTCLING1S     := $(wildcard $(UTILSDIRS)/rootcling_stage1.cxx)
ROOTCLING1O     := $(call stripsrc,$(ROOTCLING1S:.cxx=.o))
ROOTCLING1DEP    := $(ROOTCLING1O:.o=.d)

ROOTCLING1EXE   := $(call stripsrc,$(UTILSDIRS)/rootcling_stage1$(EXEEXT))
ROOTCLINGSTAGE1 := $(ROOTCLING1EXE)

##### Dependencies for all dictionaries
ROOTCLINGSTAGE1DEP := $(ROOTCLINGSTAGE1)

# include all dependency files
INCLUDEFILES += $(ROOTCLING1DEP)

ROOTCLINGCXXFLAGS = $(filter-out -fno-exceptions,$(filter-out -Wcast-qual,$(CLINGCXXFLAGS)))
ifneq ($(CXX:g++=),$(CXX))
ROOTCLINGCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif

endif # ifneq ($(HOST),)
