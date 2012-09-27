# ModuleVars.mk for utils module
# Copyright (c) 2008 Rene Brun and Fons Rademakers
#
# Author: Axel Naumann, 2008-06-10

ifneq ($(HOST),)

UTILSDIRS    := $(BUILDTOOLSDIR)/core/utils/src

ifeq ($(BUILDCLING),yes)
ROOTCLINGS    := $(UTILSDIRS)/rootcling.cxx \
                 $(filter-out %RStl.cxx,$(filter-out %root%.cxx,$(filter-out %_tmp.cxx,$(wildcard $(UTILSDIRS)/*.cxx))))
ROOTCLINGTMPS := $(ROOTCLINGS:.cxx=_tmp.cxx)
ROOTCLINGTMPO := $(ROOTCLINGS:.cxx=_tmp.o)
ROOTCLINGTMPEXE := $(UTILSDIRS)/rootcling_tmp$(EXEEXT)
ROOTCINTTMP  ?= $(ROOTCLINGTMPEXE) -$(ROOTDICTTYPE)

else 
ROOTCINTS    := $(UTILSDIRS)/rootcint.cxx \
                $(filter-out %_tmp.cxx,$(wildcard $(UTILSDIRS)/R*.cxx))
ROOTCINTTMPO := $(ROOTCINTS:.cxx=_tmp.o)
ROOTCINTTMPEXE := $(UTILSDIRS)/rootcint_tmp$(EXEEXT)
ROOTCINTTMP  ?= $(ROOTCINTTMPEXE) -$(ROOTDICTTYPE)

endif

##### Dependencies for all dictionaries
ifeq ($(BUILDCLING),yes)
ROOTCINTTMPDEP = $(ROOTCLINGTMPO) $(ORDER_) $(ROOTCLINGTMPEXE)
else
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)
endif

##### rlibmap #####
RLIBMAP      := $(BUILDTOOLSDIR)/bin/rlibmap$(EXEEXT)

else # ifneq ($(HOST),)

MODNAME      := utils

MODDIR       := $(ROOT_SRCDIR)/core/$(MODNAME)
MODDIRS      := $(MODDIR)/src
MODDIRI      := $(MODDIR)/inc

UTILSDIR     := $(MODDIR)
UTILSDIRS    := $(UTILSDIR)/src
UTILSDIRI    := $(UTILSDIR)/inc

##### rootcint #####%
ifneq ($(BUILDCLING),yes)
ROOTCINTS    := $(UTILSDIRS)/rootcint.cxx \
                $(filter-out %RClStl.cxx %_tmp.cxx,$(wildcard $(UTILSDIRS)/R*.cxx))
ROOTCINTO    := $(call stripsrc,$(ROOTCINTS:.cxx=.o))
ROOTCINTTMPO := $(call stripsrc,$(ROOTCINTS:.cxx=_tmp.o))
ROOTCINTDEP  := $(ROOTCINTO:.o=.d) $(ROOTCINTTMPO:.o=.d)

ROOTCINTTMPEXE := $(call stripsrc,$(UTILSDIRS)/rootcint_tmp$(EXEEXT))
ROOTCINTEXE  := bin/rootcint$(EXEEXT)
ROOTCINTTMP  ?= $(ROOTCINTTMPEXE) -$(ROOTDICTTYPE)
endif # ifneq ($(BUILDCLING),yes)

##### rootcling #####
ifeq ($(BUILDCLING),yes)
ROOTCLINGUTILS := $(filter-out %/rlibmap.cxx %/rootcint.cxx %/rootcling.cxx %/RStl.cxx %_tmp.cxx,\
                  $(wildcard $(UTILSDIRS)/*.cxx))
ROOTCLINGUTILO := $(call stripsrc,$(ROOTCLINGUTILS:.cxx=.o))
ROOTCLINGS := $(wildcard $(UTILSDIRS)/rootcling.cxx)
ROOTCLINGO := $(call stripsrc,$(ROOTCLINGS:.cxx=.o))
ROOTCLINGTMPS := $(call stripsrc,$(ROOTCLINGS:.cxx=_tmp.cxx))
ROOTCLINGTMPO := $(ROOTCLINGTMPS:.cxx=.o)
ROOTCLINGDEP := $(ROOTCLINGO:.o=.d) $(ROOTCLINGTMPO:.o=.d) $(ROOTCLINGUTILO:.o=.d)

ROOTCLINGTMPEXE := $(call stripsrc,$(UTILSDIRS)/rootcling_tmp$(EXEEXT))
ROOTCLINGEXE  := bin/rootcling$(EXEEXT)
ROOTCLINGTMP  ?= $(ROOTCLINGTMPEXE) -$(ROOTDICTTYPE)
endif # ifeq ($(BUILDCLING),yes)

##### Dependencies for all dictionaries
ifeq ($(BUILDCLING),yes)
ROOTCINTTMPDEP = $(ROOTCLINGTMPO) $(ORDER_) $(ROOTCLINGTMPEXE)
else
ROOTCINTTMPDEP = $(ROOTCINTTMPO) $(ORDER_) $(ROOTCINTTMPEXE)
endif

##### rlibmap #####
RLIBMAPS     := $(UTILSDIRS)/rlibmap.cxx
RLIBMAPO     := $(call stripsrc,$(RLIBMAPS:.cxx=.o))
RLIBMAPDEP   := $(RLIBMAPO:.o=.d)
RLIBMAP      := bin/rlibmap$(EXEEXT)

# include all dependency files
INCLUDEFILES += $(ROOTCINTDEP) $(ROOTCLINGDEP) $(RLIBMAPDEP)

ifeq ($(BUILDCLING),yes)
ROOTCLINGCXXFLAGS = $(filter-out -Wcast-qual,$(CLINGCXXFLAGS))
ifneq ($(CXX:g++=),$(CXX))
ROOTCLINGCXXFLAGS += -Wno-shadow -Wno-unused-parameter
endif
else # ifeq ($(BUILDCLING),yes)
ROOTCLINGCXXFLAGS := 
endif # ifeq ($(BUILDCLING),yes)

endif # ifneq ($(HOST),)
